package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tuancookiez-hub/hyatlas-v4/memory"
)

// dimProbeServer answers /embeddings with a vector of dims floats and counts the
// requests, so one test can pin both the probed value and that it was probed once.
// A dims of 0 makes it answer 500, for the "endpoint cannot be probed" branch.
func dimProbeServer(t *testing.T, dims int, calls *atomic.Int64) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls != nil {
			calls.Add(1)
		}
		w.Header().Set("Content-Type", "application/json")
		if dims <= 0 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"code":20015,"message":"probe refused"}}`))
			return
		}
		vec := make([]float32, dims)
		vec[0] = 1 // already normalized
		out, _ := json.Marshal(map[string]any{"data": []map[string]any{{"embedding": vec}}})
		_, _ = w.Write(out)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// A remote embedder reports the length its endpoint really returns. 384 here was a
// hardcoded placeholder, and the store sizes its zero-vector enumeration queries
// from this number, so a 1024d collection was queried with a 384d vector.
func TestOpenAIEmbedderProbesTheRealDimension(t *testing.T) {
	var calls atomic.Int64
	srv := dimProbeServer(t, 1024, &calls)
	e := NewOpenAIEmbedder(srv.URL, "", "BAAI/bge-large-zh-v1.5")
	if got := e.Dim(); got != 1024 {
		t.Fatalf("Dim() = %d, want the 1024 the endpoint returns", got)
	}
	if !strings.Contains(e.DimSource(), "probe") {
		t.Errorf("DimSource() = %q, want it to name the probe", e.DimSource())
	}
}

// The probe costs one request per process, not one per Dim() call: Dim() is read on
// every layer of the startup rebuild and by /api/v1/status.
func TestOpenAIEmbedderProbesOnlyOnce(t *testing.T) {
	var calls atomic.Int64
	srv := dimProbeServer(t, 768, &calls)
	e := NewOpenAIEmbedder(srv.URL, "", "m")
	for i := 0; i < 4; i++ {
		if got := e.Dim(); got != 768 {
			t.Fatalf("call %d: Dim() = %d, want 768", i, got)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("endpoint probed %d times, want exactly 1", got)
	}
}

// HYATLAS_EMBED_DIMS wins over the probe, and setting it means no probe request is
// made at all (which is the point of the override on an offline gateway).
func TestOpenAIEmbedderDimEnvOverrideWins(t *testing.T) {
	var calls atomic.Int64
	srv := dimProbeServer(t, 1024, &calls)
	t.Setenv(embedDimsEnv, "768")
	e := NewOpenAIEmbedder(srv.URL, "", "m")
	if got := e.Dim(); got != 768 {
		t.Fatalf("Dim() = %d, want the %s override 768", got, embedDimsEnv)
	}
	if got := calls.Load(); got != 0 {
		t.Errorf("endpoint probed %d times, want 0 when %s is set", got, embedDimsEnv)
	}
	if got := e.DimSource(); got != embedDimsEnv {
		t.Errorf("DimSource() = %q, want %q", got, embedDimsEnv)
	}
}

// fastProbe is the probe's timing shrunk to milliseconds, so a test that exercises the
// retry loop does not spend the default minute.
var fastProbe = probeTiming{
	total: 200 * time.Millisecond, attempt: 50 * time.Millisecond,
	backoff: 5 * time.Millisecond, backoffMax: 20 * time.Millisecond,
}

// A probe that cannot run retries (the endpoint is often another application that
// starts after this one) and only then falls back, at ERROR level, to the historical
// 384 — saying which one it used.
func TestOpenAIEmbedderDimFallsBackWhenProbeFails(t *testing.T) {
	var calls atomic.Int64
	srv := dimProbeServer(t, 0, &calls) // answers 500
	e := NewOpenAIEmbedder(srv.URL, "", "m")
	e.probe = fastProbe
	if got := e.Dim(); got != fallbackEmbedDim {
		t.Fatalf("Dim() = %d, want the %d fallback", got, fallbackEmbedDim)
	}
	if got := calls.Load(); got < 2 {
		t.Errorf("the endpoint was probed %d time(s): a failed probe must be retried before falling back", got)
	}
	if src := e.DimSource(); !strings.Contains(src, "fallback") {
		t.Errorf("DimSource() = %q, want it to say the fallback was used", src)
	}
}

// A probe that fails while the gateway is still starting must not lock the process onto
// 384: the retry loop gives the endpoint time, and the first answer decides.
func TestOpenAIEmbedderProbeRetriesUntilTheEndpointAnswers(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if n < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"error":"still starting"}`))
			return
		}
		vec := make([]float32, 1024)
		vec[0] = 1
		out, _ := json.Marshal(map[string]any{"data": []map[string]any{{"embedding": vec}}})
		_, _ = w.Write(out)
	}))
	t.Cleanup(srv.Close)

	e := NewOpenAIEmbedder(srv.URL, "", "m")
	e.probe = fastProbe
	if got := e.Dim(); got != 1024 {
		t.Fatalf("Dim() = %d, want the 1024 the endpoint answered once it was up", got)
	}
	if got := calls.Load(); got != 3 {
		t.Errorf("endpoint probed %d times, want 3 (two refusals then an answer)", got)
	}
}

func TestOpenAIEmbedderDimFallsBackToEnvWhenProbeFails(t *testing.T) {
	srv := dimProbeServer(t, 0, nil)
	t.Setenv(embedDimsEnv, "2560")
	e := NewOpenAIEmbedder(srv.URL, "", "m")
	if got := e.Dim(); got != 2560 {
		t.Fatalf("Dim() = %d, want the %s override 2560", got, embedDimsEnv)
	}
}

// A junk override must not silently resize the vectors: it is ignored and the
// endpoint is probed instead.
func TestOpenAIEmbedderDimOverrideRejectsJunk(t *testing.T) {
	var calls atomic.Int64
	srv := dimProbeServer(t, 1024, &calls)
	t.Setenv(embedDimsEnv, "not-a-number")
	e := NewOpenAIEmbedder(srv.URL, "", "m")
	if got := e.Dim(); got != 1024 {
		t.Fatalf("Dim() = %d, want the probed 1024 after ignoring the junk override", got)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("endpoint probed %d times, want 1", got)
	}
}

// The probe must not survive a wrap: the store gates its "wait for the remote
// embedder" step on this, and a decorator that hides it would remove the wait.
func TestRemoteDetectionSurvivesDecoration(t *testing.T) {
	if !isRemote(NewOpenAIEmbedder("http://127.0.0.1:1/v1", "", "m")) {
		t.Error("a bare OpenAIEmbedder must count as remote")
	}
	if isRemote(NewLocalEmbedder(384)) {
		t.Error("the local stub must not count as remote")
	}
	s := &MemoryStore{embed: NewOpenAIEmbedder("http://127.0.0.1:1/v1", "", "m")}
	if !s.needsEmbedderProbe() {
		t.Error("a remote embedder must trigger the readiness probe")
	}
}

// The live bug, end to end: a 1024d remote endpoint must give the store 1024, so the
// startup rebuild enumerates the 1024d collections with a 1024d zero vector and the
// doc index survives the restart. With the hardcoded 384 the rebuild indexed nothing
// and persistGuard refused to write doc_index.json.
func TestRemoteEmbedderDimsDriveTheStartupRebuild(t *testing.T) {
	srv := dimProbeServer(t, 1024, nil)
	dir := t.TempDir()
	graphPath := filepath.Join(dir, "graph.json")
	indexPath := filepath.Join(dir, "doc_index.json")
	e := NewOpenAIEmbedder(srv.URL, "", "BAAI/bge-large-zh-v1.5")

	s1, err := NewMemoryStore(ctxForTest(), dir, e, graphPath, e.Dim())
	if err != nil {
		t.Skipf("MemoryStore unavailable in test env: %v", err)
	}
	id := newID()
	if err := s1.Add(memory.L3Fact, id, "a short fact", map[string]string{"user_id": "u", "agent_id": "a"}); err != nil {
		t.Fatalf("add: %v", err)
	}
	s1.Close()

	// Reopen: this is the rebuildIndex path that logged "zero-vector query failed
	// ... (query dim=384)" in production.
	e2 := NewOpenAIEmbedder(srv.URL, "", "BAAI/bge-large-zh-v1.5")
	s2, err := NewMemoryStore(ctxForTest(), dir, e2, graphPath, e2.Dim())
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer s2.Close()
	if got := s2.TotalMemories(); got != 1 {
		t.Errorf("reopened store holds %d memories, want 1", got)
	}
	if got := countIndexOnDisk(t, indexPath); got != 1 {
		t.Errorf("doc_index.json holds %d entries after the rebuild, want 1 "+
			"(0 means persistGuard refused the overwrite, the production symptom)", got)
	}
}
