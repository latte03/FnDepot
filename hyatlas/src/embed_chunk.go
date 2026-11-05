package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"unicode/utf8"
)

// A remote embedding model rejects an over-long input with HTTP 400. The fork hit
// that in production: a ~1500-rune memory was sent whole, the endpoint answered
// 400/code 20015, the store wrapped it as "couldn't create embedding of document"
// and /api/v1/add answered 500 — the memory was simply lost.
//
// The limit is tokens, not characters, so no character budget is safe: on
// siliconflow/BAAI/bge-large-zh-v1.5 (the live NAS model) one 510-rune text passed
// while a different 511-rune text was rejected, i.e. the boundary moves with the
// content (511..929 runes observed). Chunking is therefore a heuristic first step,
// not an assumption: whatever is sent may still come back 400, and that must be
// retried smaller rather than surfaced as a failure.
//
// ChunkedEmbedder is installed once, around the configured embedder, so every
// embedding call site (writes, clause rows, L5 search documents, entity pages,
// queries) is covered: a text longer than chunkChars is cut into pieces at
// sentence/space boundaries, each piece is embedded, and the piece vectors are
// mean-pooled and L2-normalized back into the single vector chromem requires.
//
// Because that 400 is not length-specific (the same provider answers 400/code 20015
// for an empty or a missing "input"), the halving retries are charged against one
// whole-call budget and the call gives up with the provider's original error: the
// worst-case work per Embed is bounded, whatever the real cause was.
type ChunkedEmbedder struct {
	inner Embedder
	// chunkChars is the rune budget one request aims at. It is deliberately below
	// the smallest observed rejection point, so the first attempt usually succeeds.
	chunkChars int
}

const (
	// embedChunkCharsEnv sets that rune budget. It exists for a model whose limit is
	// far from the live one, not to turn chunking off: a value that is not a positive
	// integer falls back to the default instead, because switching the splitter off
	// reintroduces the 500 this decorator exists to prevent.
	embedChunkCharsEnv = "HYATLAS_EMBED_CHUNK_CHARS"
	// defaultEmbedChunkChars is a safe first attempt for the live model, which
	// rejected a 511-rune text: half of it leaves room for a token-dense text.
	defaultEmbedChunkChars = 400
	// embedChunkMax caps the pieces one text may be cut into, so a pathologically long
	// text fails with a clear error instead of a request storm.
	//
	// 80 is not arbitrary: extractMaxChars (server.go) is the longest memory the
	// extractor accepts, and embedChunkMax × chunkChars must cover it — 30000/400 = 75
	// pieces, plus headroom. At the default 400 that is 32,000 runes. A smaller
	// HYATLAS_EMBED_CHUNK_CHARS shrinks that cover, which is exactly the mismatch
	// embedChunkCapWarning reports at startup, so the two cannot disagree silently.
	//
	// Worst-case work per Embed stays finite: 80 pieces × at most 1+2+4 requests along
	// one descent ≈ 560 requests, and the halving budget caps it further.
	embedChunkMax = 80
	// embedChunkMinRunes is the size below which a rejected piece is not split again:
	// a smaller request cannot be the fix for a rejection at this size.
	embedChunkMinRunes = 64
	// embedSplitBudget is the floor of one Embed call's halving budget; the real budget
	// is per piece (see splitBudget). It is shared by the whole call on purpose: a
	// per-piece budget multiplies (each half may split again), and a 400 is not always
	// about length — an empty or missing "input" answers the same 400 as an over-long
	// one, and that must not turn into an ever-shrinking retry storm. When the budget is
	// gone the call fails with the original rejection.
	embedSplitBudget = 3
)

// splitBudget is the halving budget of one Embed call: enough for every piece to be
// reduced all the way from the chunk budget down to the split floor, capped at the piece
// cap, and never below the single-piece constant.
//
// One piece needs at most ceil(log2(chunkChars/minRunes)) halvings (400/64 → 3), so the
// whole call needs at most pieces × that. The work per call is therefore finite whatever
// the endpoint answers (a piece costs at most 1+2+4 requests along one descent), and yet
// a long token-dense text — the input this decorator exists for — is not starved by the
// pieces before it. A flat constant starved exactly that case: measured with a stub whose
// limit forced two halvings per piece, four pieces needed ~8 halvings and died at 3.
func (e *ChunkedEmbedder) splitBudget(pieces int) int {
	if pieces < 1 {
		pieces = 1
	}
	depth := 1
	if e.chunkChars > embedChunkMinRunes {
		depth = int(math.Ceil(math.Log2(float64(e.chunkChars) / float64(embedChunkMinRunes))))
	}
	n := pieces * depth
	if n > embedChunkMax {
		n = embedChunkMax
	}
	if n < embedSplitBudget {
		n = embedSplitBudget
	}
	return n
}

// embedChunkCapWarning reports a chunk configuration that cannot hold the longest memory
// the extractor accepts: embedChunkMax × chunkChars must cover extractMaxChars, or a text
// the extractor is happy to take can never be stored (it fails with the "over the
// N-chunk cap" error). Empty means the two agree.
//
// extractMaxChars is a byte cap and chunkChars is a rune budget, so comparing them
// directly is conservative for non-ASCII text: a rune is at least one byte, so no text
// that fits the byte cap needs more runes than the cap.
func embedChunkCapWarning(chunkChars int) string {
	if chunkChars <= 0 {
		return ""
	}
	need := (extractMaxChars + chunkChars - 1) / chunkChars
	if need <= embedChunkMax {
		return ""
	}
	minChars := (extractMaxChars + embedChunkMax - 1) / embedChunkMax
	return fmt.Sprintf("WARNING hyatlas: %s=%d needs %d chunks for a %d-character memory (the extraction limit), "+
		"but the chunk cap is %d: a memory that long cannot be stored. Raise %s to at least %d, or shorten the memory",
		embedChunkCharsEnv, chunkChars, need, extractMaxChars, embedChunkMax, embedChunkCharsEnv, minChars)
}

// resolveEmbedChunkChars reads HYATLAS_EMBED_CHUNK_CHARS, defaulting to
// defaultEmbedChunkChars. Junk and non-positive values are reported and replaced by
// the default (see embedChunkCharsEnv).
func resolveEmbedChunkChars() int {
	raw := strings.TrimSpace(os.Getenv(embedChunkCharsEnv))
	if raw == "" {
		return defaultEmbedChunkChars
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		log.Printf("%s=%q is not a positive integer; using %d", embedChunkCharsEnv, raw, defaultEmbedChunkChars)
		return defaultEmbedChunkChars
	}
	return n
}

// NewChunkedEmbedder wraps inner so no single embedding request carries more than
// chunkChars runes. A non-positive chunkChars means the default.
func NewChunkedEmbedder(inner Embedder, chunkChars int) *ChunkedEmbedder {
	if chunkChars <= 0 {
		chunkChars = defaultEmbedChunkChars
	}
	return &ChunkedEmbedder{inner: inner, chunkChars: chunkChars}
}

// embedCall is the state of one Embed call: how long the text the caller asked about
// is, how many halvings are left, and the first rejection seen. That first error is
// what a failure reports — the deepest piece's error would only say that a 64-rune
// fragment was refused, which is not what the caller needs to know.
type embedCall struct {
	total int
	left  int
	first error
	piece int
}

// note records a rejection: the first one is kept for the final report, and the piece
// size is tracked so the report can say how far the text was already reduced.
func (c *embedCall) note(err error, piece int) {
	if c.first == nil {
		c.first = err
	}
	if c.piece == 0 || piece < c.piece {
		c.piece = piece
	}
}

// spend charges one halving against the call's budget, reporting whether it was
// available. The budget is shared by every piece of the text (see embedSplitBudget).
func (c *embedCall) spend() bool {
	if c.left <= 0 {
		return false
	}
	c.left--
	return true
}

// Embed embeds text as one request when it fits the rune budget, and otherwise as
// chunks pooled back into one vector. A rejection of the whole text — including one
// that fits the budget, since the budget is only a guess at a token limit — falls
// through to the same splitting, so a short text is never failed on a length error.
func (e *ChunkedEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	total := utf8.RuneCountInString(text)
	if total <= e.chunkChars {
		c := &embedCall{total: total, left: e.splitBudget(1)}
		v, err := e.inner.Embed(ctx, text)
		if err == nil {
			return v, nil
		}
		if !looksLikeInputTooLong(err) {
			return nil, e.lengthErr(total, total, err)
		}
		c.note(err, total)
		if total <= embedChunkMinRunes || !c.spend() {
			return nil, e.giveUp(c)
		}
		// The text fits the rune budget but the model still refused it: the budget is
		// not a safe budget for this content, so halve instead of failing. The request
		// that failed is not repeated.
		return e.embedHalves(ctx, text, c)
	}
	pieces := splitEmbedChunks(text, e.chunkChars)
	if len(pieces) > embedChunkMax {
		return nil, fmt.Errorf("text of %d runes needs %d chunks of at most %d runes (model %s), "+
			"over the %d-chunk cap; raise %s or shorten the memory",
			total, len(pieces), e.chunkChars, e.modelName(), embedChunkMax, embedChunkCharsEnv)
	}
	// The budget scales with the piece count (see splitBudget): every piece may need
	// one halving without starving the pieces after it.
	c := &embedCall{total: total, left: e.splitBudget(len(pieces))}
	vecs := make([][]float32, 0, len(pieces))
	for _, p := range pieces {
		v, err := e.embedPiece(ctx, p, c)
		if err != nil {
			return nil, err
		}
		vecs = append(vecs, v)
	}
	return poolVectors(vecs)
}

// embedPiece embeds one chunk, halving it while the call still has budget when the
// model rejects it as too long. Any other rejection is reported as-is: it is not
// something a shorter input can fix.
func (e *ChunkedEmbedder) embedPiece(ctx context.Context, piece string, c *embedCall) ([]float32, error) {
	v, err := e.inner.Embed(ctx, piece)
	if err == nil {
		return v, nil
	}
	n := utf8.RuneCountInString(piece)
	if !looksLikeInputTooLong(err) {
		return nil, e.lengthErr(c.total, n, err)
	}
	c.note(err, n)
	if n <= embedChunkMinRunes || !c.spend() {
		return nil, e.giveUp(c)
	}
	return e.embedHalves(ctx, piece, c)
}

// embedHalves cuts piece in two at a boundary near its middle and pools the halves,
// each of which is itself retried smaller. The piece is already known to have been
// rejected, so this does not repeat the request that failed.
func (e *ChunkedEmbedder) embedHalves(ctx context.Context, piece string, c *embedCall) ([]float32, error) {
	left, right := halfRunes(piece)
	if left == "" || right == "" {
		return nil, e.giveUp(c)
	}
	vl, err := e.embedPiece(ctx, left, c)
	if err != nil {
		return nil, err
	}
	vr, err := e.embedPiece(ctx, right, c)
	if err != nil {
		return nil, err
	}
	return poolVectors([][]float32{vl, vr})
}

// giveUp reports the call as failed, keeping the provider's own error text (the first
// one seen) rather than only the size of the last fragment that happened to be tried.
func (e *ChunkedEmbedder) giveUp(c *embedCall) error {
	if c.first == nil {
		return e.lengthErr(c.total, c.piece, fmt.Errorf("the text could not be split into acceptable pieces"))
	}
	return e.lengthErr(c.total, c.piece, c.first)
}

// lengthErr annotates a rejection with the facts the provider leaves out and the
// caller needs: how many runes were submitted, which model refused them, and how far
// the piece was already reduced (upstream issue #24 is a bare HTTP 400 surfacing as a
// 500 on the write path). Only the level where the rejection happened wraps the error,
// so the message is not repeated once per halving.
func (e *ChunkedEmbedder) lengthErr(total, piece int, err error) error {
	return fmt.Errorf("couldn't embed %d rune(s) with model %s "+
		"(smallest rejected piece %d runes, chunk budget %d from %s, split floor %d, max %d chunks): %w",
		total, e.modelName(), piece, e.chunkChars, embedChunkCharsEnv, embedChunkMinRunes, embedChunkMax, err)
}

// modelName names the model behind the vectors, for those messages.
func (e *ChunkedEmbedder) modelName() string { return embedderModelName(e.inner) }

// Dim forwards the wrapped embedder's dimension: chunking does not change the vector
// length, and the store sizes its zero-vector queries from this number.
func (e *ChunkedEmbedder) Dim() int { return e.inner.Dim() }

// DimSource forwards the wrapped embedder's explanation of where that dimension came
// from, so the startup log still names the probe/env/fallback behind the wrapper.
func (e *ChunkedEmbedder) DimSource() string { return embedDimSource(e.inner) }

// ModelName names the model behind the vectors, so a length error can say what
// refused the text.
func (e *ChunkedEmbedder) ModelName() string { return e.modelName() }

// Semantic passes on whether the wrapped embedder's vectors carry meaning: pooling
// does not make a hashed-local stub semantic.
func (e *ChunkedEmbedder) Semantic() bool {
	if s, ok := e.inner.(interface{ Semantic() bool }); ok {
		return s.Semantic()
	}
	return true
}

// Unwrap exposes the wrapped embedder, so capability checks (isRemote,
// needsEmbedderProbe) still see through the decorator instead of silently answering
// "not remote" and skipping the readiness wait.
func (e *ChunkedEmbedder) Unwrap() Embedder { return e.inner }

// embedderModelName names the model behind an embedder, for error messages. An
// embedder that cannot name itself reports "unknown" rather than nothing.
func embedderModelName(e Embedder) string {
	if m, ok := e.(interface{ ModelName() string }); ok {
		if n := strings.TrimSpace(m.ModelName()); n != "" {
			return n
		}
	}
	if m, ok := e.(interface{ Unwrap() Embedder }); ok {
		return embedderModelName(m.Unwrap())
	}
	return "unknown"
}

// inputTooLongHints are body/status substrings that mark a rejection as "the input was
// too long". The stable signal is the status prefix ("embed HTTP 400": the embedder's
// own message form, which survives body truncation), so these are only extra evidence.
// Some of them are wide ("exceeds", "too large") and can match an unrelated error; that
// is acceptable because this only chooses which path to try — the work is bounded by
// splitBudget, so a misclassification costs a few requests, not a retry storm.
//
// A provider's private code is deliberately not relied on: the fork saw SiliconFlow
// answer 400/code 20015 for an over-long input, but the error carries only the first
// 200 bytes of the body and the nested code falls outside that window — and the same
// 400/20015 is what that provider returns for an empty or missing "input" too.
var inputTooLongHints = []string{
	"embed http 400", "embed http 413",
	"too long", "too large", "too many token", "max token", "maximum token",
	"context length", "input length", "token limit", "exceeds",
}

// looksLikeInputTooLong reports whether err is worth answering by sending a shorter
// input. Because a 400 is not length-specific, the answer is always bounded by
// embedSplitBudget: this decides which path to try, not how long to keep trying.
// A dimension mismatch from chromem also mentions "length", so it is excluded
// explicitly — bisecting the text cannot fix a vector-size disagreement.
func looksLikeInputTooLong(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	if strings.Contains(msg, "vectors must have the same length") {
		return false
	}
	for _, h := range inputTooLongHints {
		if strings.Contains(msg, h) {
			return true
		}
	}
	return false
}

// isEmbedBreakRune reports whether a rune is a boundary worth cutting at: a line or
// paragraph end, whitespace, or a sentence/clause mark in the scripts the store sees.
func isEmbedBreakRune(r rune) bool {
	switch r {
	case '\n', '\r', '\t', ' ', '。', '！', '？', '；', '，', '、', ';', ',', '.', '!', '?':
		return true
	}
	return false
}

// splitEmbedChunks cuts text into pieces of at most max runes, preferring a cut just
// after a break rune in the second half of the window so pieces stay close to max.
// The pieces concatenate back to text (the break rune stays with the piece it ends),
// so pooling can never silently drop input.
func splitEmbedChunks(text string, max int) []string {
	if max <= 0 {
		return []string{text}
	}
	runes := []rune(text)
	var out []string
	for len(runes) > max {
		cut := max
		for i := max - 1; i >= max/2; i-- {
			if isEmbedBreakRune(runes[i]) {
				cut = i + 1
				break
			}
		}
		out = append(out, string(runes[:cut]))
		runes = runes[cut:]
	}
	if len(runes) > 0 {
		out = append(out, string(runes))
	}
	return out
}

// halfRunes splits a piece in two, preferring a break rune near the middle so each
// half still ends at a boundary. An empty half means the piece cannot be split.
func halfRunes(piece string) (string, string) {
	runes := []rune(piece)
	if len(runes) < 2 {
		return "", ""
	}
	mid := len(runes) / 2
	cut := mid
	for i := mid; i > 0; i-- {
		if isEmbedBreakRune(runes[i]) {
			cut = i + 1
			break
		}
	}
	if cut <= 0 || cut >= len(runes) {
		cut = mid
	}
	return string(runes[:cut]), string(runes[cut:])
}

// poolVectors mean-pools chunk vectors and L2-normalizes the result. Normalizing is
// not cosmetic: chromem compares every document by dot product and only normalizes
// vectors that arrive unnormalized, so a pooled vector must already be unit length.
func poolVectors(vecs [][]float32) ([]float32, error) {
	if len(vecs) == 0 {
		return nil, fmt.Errorf("no chunk vectors to pool")
	}
	dim := len(vecs[0])
	if dim == 0 {
		return nil, fmt.Errorf("chunk 0 came back with an empty vector")
	}
	out := make([]float32, dim)
	for i, v := range vecs {
		if len(v) != dim {
			return nil, fmt.Errorf("chunk %d has %d dimensions but chunk 0 has %d; "+
				"the endpoint changed vector length mid-text", i, len(v), dim)
		}
		for j, x := range v {
			out[j] += x
		}
	}
	inv := float32(1) / float32(len(vecs))
	for j := range out {
		out[j] *= inv
	}
	return normalize(out), nil
}
