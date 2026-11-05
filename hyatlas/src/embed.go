package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	bgeemb "github.com/tuancookiez-hub/hyatlas-v4/bge"
)

// Embedder produces normalized vectors (chromem requires normalized).
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	// Dim reports the embedding dimension this embedder produces. Fork (fnos):
	// the fork ships bge-large-zh (1024d), so /api/v1/status must report what
	// the store actually indexes rather than the historical 384 default. The
	// store captures this at open time and reports it via embedDims; tests that
	// wrap another embedder (flakyEmbedder, blockingEmbedder) forward Dim()
	// so the interface stays uniform.
	Dim() int
}

// OpenAIEmbedder calls an OpenAI-compatible /embeddings endpoint.
// Routes through the ai2api proxy on loopback (firewall-safe) by default.
type OpenAIEmbedder struct {
	BaseURL string
	APIKey  string
	Model   string
	Client  *http.Client

	// dimOnce runs the single probe that learns the endpoint's vector length, so a
	// start costs one extra request instead of one per Dim() call. dim stays 0
	// until it has run; dimSrc names where the value came from, for the startup log.
	dimOnce sync.Once
	dim     int
	dimSrc  string
	// probe bounds that probe's retries. The zero value means defaultProbeTiming;
	// tests shrink it so a refused endpoint does not cost a minute.
	probe probeTiming
}

func NewOpenAIEmbedder(baseURL, key, model string) *OpenAIEmbedder {
	return &OpenAIEmbedder{BaseURL: baseURL, APIKey: key, Model: model,
		Client: &http.Client{Timeout: 60 * time.Second}}
}

func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	body, _ := json.Marshal(map[string]any{"model": e.Model, "input": []string{text}})
	req, err := http.NewRequestWithContext(ctx, "POST", e.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.APIKey)
	}
	resp, err := e.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("embed HTTP %d (model %s, %d runes): %s",
			resp.StatusCode, e.Model, len([]rune(text)), truncStr(data, 200))
	}
	var out struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	if len(out.Data) == 0 {
		return nil, fmt.Errorf("embed: no data returned")
	}
	return normalize(out.Data[0].Embedding), nil
}

// fallbackEmbedDim is the vector length assumed when the endpoint cannot be probed
// and HYATLAS_EMBED_DIMS is unset. It is the upstream default (BGE-small, which is
// also the shape of the text-embedding-3-small endpoint the fork inherited), kept so
// an unreachable endpoint sizes vectors exactly as it did before probing existed.
const fallbackEmbedDim = 384

// embedDimsEnv overrides the embedder's vector length. Set it when the endpoint
// cannot answer a probe (an offline gateway, a proxy that drops /embeddings), or to
// skip the probe request entirely.
const embedDimsEnv = "HYATLAS_EMBED_DIMS"

// embedProbeText is the shortest text that still makes a real embedding request; the
// response's vector length is all Dim() needs from it.
const embedProbeText = "dim"

// probeTiming bounds the dimension probe. The zero value means defaultProbeTiming;
// tests shrink it so a refused endpoint does not cost a minute.
type probeTiming struct {
	total      time.Duration // whole budget, all attempts together
	attempt    time.Duration // one request
	backoff    time.Duration // first wait between attempts
	backoffMax time.Duration // wait ceiling
}

// defaultProbeTiming mirrors waitForEmbedder's readiness wait (500ms doubling to 5s,
// about a minute in total). The endpoint is often another application that starts after
// this one — on the NAS the gateway comes up second — and a single failed probe used to
// fall straight back to 384, which put the store back on 384d zero-vector queries
// against 1024d documents: the very bug this fix removes, just with another trigger.
var defaultProbeTiming = probeTiming{
	total:      60 * time.Second,
	attempt:    10 * time.Second,
	backoff:    500 * time.Millisecond,
	backoffMax: 5 * time.Second,
}

// orDefault fills in whatever the caller left at zero.
func (p probeTiming) orDefault() probeTiming {
	d := defaultProbeTiming
	if p.total > 0 {
		d.total = p.total
	}
	if p.attempt > 0 {
		d.attempt = p.attempt
	}
	if p.backoff > 0 {
		d.backoff = p.backoff
	}
	if p.backoffMax > 0 {
		d.backoffMax = p.backoffMax
	}
	return d
}

// embedDimsOverride returns HYATLAS_EMBED_DIMS when it holds a positive integer, else
// 0 ("unset"). A junk value is reported once and ignored, so a typo cannot silently
// size every vector the store writes.
func embedDimsOverride() int {
	raw := strings.TrimSpace(os.Getenv(embedDimsEnv))
	if raw == "" {
		return 0
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		log.Printf("%s=%q is not a positive integer; ignoring it and probing the endpoint", embedDimsEnv, raw)
		return 0
	}
	return n
}

// probeDim asks the endpoint for one short embedding and reports its length, retrying
// with a bounded backoff (see defaultProbeTiming). Only when the whole budget is spent
// does it report a failure; Dim() then falls back and says so at ERROR level.
func (e *OpenAIEmbedder) probeDim() (int, error) {
	t := e.probe.orDefault()
	deadline := time.Now().Add(t.total)
	backoff := t.backoff
	for attempt := 1; ; attempt++ {
		ctx, cancel := context.WithTimeout(context.Background(), t.attempt)
		v, err := e.Embed(ctx, embedProbeText)
		cancel()
		if err == nil {
			if len(v) == 0 {
				return 0, fmt.Errorf("the endpoint returned an empty vector for %q", embedProbeText)
			}
			if attempt > 1 {
				log.Printf("embed: dimension probe succeeded on attempt %d", attempt)
			}
			return len(v), nil
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return 0, fmt.Errorf("no answer in %s over %d attempt(s): %w", t.total, attempt, err)
		}
		wait := backoff
		if wait > remaining {
			wait = remaining
		}
		log.Printf("INFO hyatlas: waiting %v for the embedder to answer a dimension probe: %v", wait, err)
		time.Sleep(wait)
		backoff *= 2
		if backoff > t.backoffMax {
			backoff = t.backoffMax
		}
	}
}

// Dim reports the vector length the endpoint actually returns.
//
// Fork (fnos): this was a hardcoded 384. The fork ships bge-large-zh (1024d) as its
// local model and an OpenAI-compatible endpoint as its other supported mode, and the
// store sizes its zero-vector enumeration queries from this number, so a wrong answer
// is not cosmetic: every startup query compared a 384d zero vector against 1024d
// documents ("vectors must have the same length"), every layer failed, each one fell
// back to a real API call, and persistGuard could then refuse to write doc_index.json
// at all. /api/v1/status reported embed_dims=384 for the same reason.
//
// Resolution order: HYATLAS_EMBED_DIMS (explicit override, no probe), then a bounded
// live probe, then fallbackEmbedDim. The chosen value and its source are logged, so the
// startup line says where the number came from instead of leaving a mismatch to be
// discovered later in a WARNING.
func (e *OpenAIEmbedder) Dim() int {
	e.dimOnce.Do(func() {
		if n := embedDimsOverride(); n > 0 {
			e.dim, e.dimSrc = n, embedDimsEnv
			log.Printf("embed: dims=%d from %s (endpoint not probed)", n, embedDimsEnv)
			return
		}
		n, err := e.probeDim()
		if err != nil {
			e.dim, e.dimSrc = fallbackEmbedDim, "fallback: probe failed, "+embedDimsEnv+" unset"
			// ERROR, not Printf: this is a degradation, not a note, and the operator is
			// the only one who can fix it. Say what it costs, so the line is actionable:
			// on a store whose documents are not 384d, every layer of this start's index
			// rebuild fails its zero-vector query and falls back to one real API call
			// each, and /api/v1/status then reports a dimension the store does not index.
			log.Printf("ERROR hyatlas: embed dimension probe of %s failed (%v); assuming %d — "+
				"set %s (e.g. %s=1024) to name the real length. Until then this start's index "+
				"rebuild degrades to the per-layer fallback (one extra API call per layer) and "+
				"/api/v1/status reports the wrong embed_dims",
				e.BaseURL+"/embeddings", err, fallbackEmbedDim, embedDimsEnv, embedDimsEnv)
			return
		}
		e.dim, e.dimSrc = n, "probed from "+e.BaseURL+"/embeddings"
		log.Printf("embed: probed dims=%d from %s (model %s)", n, e.BaseURL+"/embeddings", e.Model)
	})
	return e.dim
}

// DimSource names where Dim() got its value, for the startup log. It runs the probe
// first, so the string is always filled by the time it returns.
func (e *OpenAIEmbedder) DimSource() string {
	e.Dim()
	return e.dimSrc
}

// ModelName names the model that produced the vectors, so an error about a text's
// length can say which model refused it (upstream issue #24: a 400 with a bare code
// and no length, on a write path that answers 500).
func (e *OpenAIEmbedder) ModelName() string { return e.Model }

// isRemote reports whether e's vectors come from another process, whose startup can
// race with ours, so the store must wait for it before it gives up a layer. A
// decorator (ChunkedEmbedder) answers for the embedder it wraps, so wrapping cannot
// hide remote-ness from that readiness gate.
func isRemote(e Embedder) bool {
	switch t := e.(type) {
	case *OpenAIEmbedder:
		return t != nil
	case interface{ Unwrap() Embedder }:
		return isRemote(t.Unwrap())
	}
	return false
}

// embedDimSource names where an embedder's dimension came from, for the startup log.
// An embedder that cannot say reports "unknown".
func embedDimSource(e Embedder) string {
	if s, ok := e.(interface{ DimSource() string }); ok {
		if v := strings.TrimSpace(s.DimSource()); v != "" {
			return v
		}
	}
	return "unknown"
}

// dimMismatchHint explains a zero-vector query that failed on vector length when the
// operator set HYATLAS_EMBED_DIMS by hand: that override is then the first thing to
// suspect, because it is the only place the store's dimension can disagree with the
// vectors already on disk. Empty when no override is set (nothing to blame).
func dimMismatchHint() string {
	raw := strings.TrimSpace(os.Getenv(embedDimsEnv))
	if raw == "" {
		return ""
	}
	return " — " + embedDimsEnv + "=" + raw + " is set: if the vectors on disk have another length, " +
		"that override is the mismatch (unset it to probe the endpoint, or set it to the stored length)"
}

// LocalEmbedder is a deterministic fallback (BGE-ONNX wired later). Normalized.
type LocalEmbedder struct{ dim int }

func NewLocalEmbedder(dim int) *LocalEmbedder { return &LocalEmbedder{dim: dim} }

// Semantic reports false: the vectors hash bytes, so their similarity says
// nothing about what two texts mean.
func (e *LocalEmbedder) Semantic() bool { return false }

// isSemantic reports whether e's vectors carry meaning. An embedder that knows
// they do not says so with a Semantic method; any other is taken at its word.
func isSemantic(e Embedder) bool {
	if s, ok := e.(interface{ Semantic() bool }); ok {
		return s.Semantic()
	}
	return true
}

func (e *LocalEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	v := make([]float32, e.dim)
	for i, b := range []byte(text) {
		v[i%e.dim] += float32(b)
	}
	return normalize(v), nil
}

func (e *LocalEmbedder) Dim() int { return e.dim }

// DimSource reports the stub's fixed dimension as its own source: nothing is probed,
// so a startup log can say that without asking the network.
func (e *LocalEmbedder) DimSource() string { return "local stub (fixed dimension)" }

// ModelName names the stub, so a length error can say what produced the vectors.
func (e *LocalEmbedder) ModelName() string { return "local-stub" }

func normalize(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	if n == 0 {
		return v
	}
	r := float32(1 / math.Sqrt(float64(n)))
	for i := range v {
		v[i] *= r
	}
	return v
}

// BGEGoEmbedder wraps the in-Go BGE model (bgeemb.BGE) as a chromem Embedder.
type BGEGoEmbedder struct {
	bge *bgeemb.BGE
}

func NewBGEGoEmbedder(modelDir, modelName string) (*BGEGoEmbedder, error) {
	m, err := bgeemb.New(modelDir, modelName)
	if err != nil {
		return nil, err
	}
	return &BGEGoEmbedder{bge: m}, nil
}

func (e *BGEGoEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return e.bge.Embed(ctx, text)
}

// Dim reports the embedding dimension the wrapped BGE model produces. Fork
// (fnos): the fork ships bge-large-zh (1024d), so /api/v1/status must report
// what the store actually indexes rather than the historical 384 default.
func (e *BGEGoEmbedder) Dim() int {
	if e.bge == nil {
		return 0
	}
	return e.bge.HiddenDim()
}

// DimSource reports the in-process model as the source of the dimension: the value
// comes from the loaded ONNX session, not from a request.
func (e *BGEGoEmbedder) DimSource() string { return "in-process bge model" }

// ModelName names the in-process model, so a length error can say what refused a text.
func (e *BGEGoEmbedder) ModelName() string { return "bge-in-process" }

func (e *BGEGoEmbedder) Destroy() {
	if e.bge != nil {
		e.bge.Destroy()
	}
}
