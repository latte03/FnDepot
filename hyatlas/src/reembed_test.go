package main

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tuancookiez-hub/hyatlas-v4/memory"
)

// oneHotEmbedder returns a unit vector with a 1 at pos, so a test can tell exactly
// which embedder (or which model) produced a stored vector.
type oneHotEmbedder struct {
	pos   int
	dim   int
	calls int
}

func (e *oneHotEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	e.calls++
	v := make([]float32, e.dim)
	v[e.pos] = 1
	return v, nil
}

func (e *oneHotEmbedder) Dim() int          { return e.dim }
func (e *oneHotEmbedder) ModelName() string { return "one-hot" }

// storedVector reads one document's vector back out of the collection, which is the
// only place a re-embed pass changes anything.
func storedVector(t *testing.T, st *MemoryStore, col, id string) []float32 {
	t.Helper()
	c := st.byName[col]
	if c == nil {
		t.Fatalf("no collection %s", col)
	}
	doc, err := c.GetByID(context.Background(), id)
	if err != nil {
		t.Fatalf("read %s/%s: %v", col, id, err)
	}
	return doc.Embedding
}

// The single source of truth must name every collection the store really creates —
// the list the offline tool carried was missing l8_entity (the entity pages) and
// l3_clause (the clause rows) entirely.
func TestAllCollectionsMatchesTheStore(t *testing.T) {
	dir := t.TempDir()
	st, err := NewMemoryStore(ctxForTest(), dir, NewLocalEmbedder(384),
		addGraphPath(dir), 384)
	if err != nil {
		t.Skipf("MemoryStore unavailable in test env: %v", err)
	}
	defer st.Close()

	want := allCollections()
	if len(want) != len(memory.All())+1 {
		t.Errorf("allCollections has %d names, want the %d layers plus l3_clause", len(want), len(memory.All()))
	}
	for _, name := range []string{string(memory.L8Entity), clauseCollection, string(memory.L3Fact)} {
		if !slices.Contains(want, name) {
			t.Errorf("allCollections is missing %s", name)
		}
	}
	if len(st.byName) != len(want) {
		t.Errorf("the store created %d collections but allCollections names %d", len(st.byName), len(want))
	}
	for _, name := range want {
		if st.byName[name] == nil {
			t.Errorf("the store did not create %s", name)
		}
	}
	// The layer map and the clause handle must be the very same collection objects,
	// or one of them would write somewhere nothing else reads.
	for _, l := range memory.All() {
		if st.cols[l] != st.byName[string(l)] {
			t.Errorf("cols[%s] is not the collection allCollections names", l)
		}
	}
	if st.clauses != st.byName[clauseCollection] {
		t.Error("the clause handle is not the collection allCollections names")
	}
}

// A dry run embeds every row (that is how it reports whether the pass would work) and
// changes nothing on disk.
func TestReembedDryRunWritesNothing(t *testing.T) {
	dir := t.TempDir()
	before := &oneHotEmbedder{pos: 0, dim: 8}
	st := newReembedStore(t, dir, before)
	id := mkFact(t, st, "alice", "a fact")
	want := storedVector(t, st, string(memory.L3Fact), id)

	rep := st.ReembedAll(context.Background(), allCollections(), 0, true)
	if rep.DryRun != true {
		t.Error("a dry run must report dry_run=true")
	}
	if rep.Done == 0 {
		t.Fatalf("dry run embedded nothing: %+v", rep)
	}
	if rep.Failed != 0 {
		t.Errorf("dry run failed %d rows: %v", rep.Failed, rep.Errors)
	}
	if got := storedVector(t, st, string(memory.L3Fact), id); !slices.Equal(got, want) {
		t.Errorf("a dry run rewrote the vector: %v -> %v", want, got)
	}
}

// A real pass replaces the vectors in place: the row keeps its id, content and
// metadata, and now carries a vector from the new embedder.
func TestReembedReplacesVectorsInPlace(t *testing.T) {
	dir := t.TempDir()
	st := newReembedStore(t, dir, &oneHotEmbedder{pos: 0, dim: 8})
	id := newID()
	meta := map[string]string{"user_id": "alice", "agent_id": "a", "session_id": "s1"}
	if err := st.Add(memory.L3Fact, id, "a fact", meta); err != nil {
		t.Fatal(err)
	}
	if got := storedVector(t, st, string(memory.L3Fact), id); got[0] != 1 {
		t.Fatalf("vector before the pass = %v, want the old embedder's", got)
	}

	after := &oneHotEmbedder{pos: 3, dim: 8}
	st.embed = after
	rep := st.ReembedAll(context.Background(), []string{string(memory.L3Fact)}, 0, false)
	if rep.Failed != 0 || rep.Done == 0 {
		t.Fatalf("pass = %+v, want no failures and re-embedded rows", rep)
	}
	got := storedVector(t, st, string(memory.L3Fact), id)
	if len(got) != 8 || got[3] != 1 || got[0] != 0 {
		t.Errorf("vector after the pass = %v, want the new embedder's", got)
	}
	row := st.GetMany([]string{id})
	if len(row) != 1 || row[0].Content != "a fact" {
		t.Errorf("the pass changed the row itself: %+v", row)
	}
	if row[0].Meta["session_id"] != "s1" {
		t.Errorf("the pass dropped metadata: %v", row[0].Meta)
	}
	if after.calls == 0 {
		t.Error("the pass never called the new embedder")
	}
}

// Running it twice must be safe and must leave the same result: overwriting a row
// with the same id is chromem's semantics, and nothing here deletes.
func TestReembedIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	st := newReembedStore(t, dir, &oneHotEmbedder{pos: 0, dim: 8})
	id := mkFact(t, st, "alice", "a fact to re-embed twice")
	st.embed = &oneHotEmbedder{pos: 5, dim: 8}

	first := st.ReembedAll(context.Background(), allCollections(), 0, false)
	one := storedVector(t, st, string(memory.L3Fact), id)
	second := st.ReembedAll(context.Background(), allCollections(), 0, false)
	two := storedVector(t, st, string(memory.L3Fact), id)

	if first.Failed != 0 || second.Failed != 0 {
		t.Fatalf("passes reported failures: %v / %v", first.Errors, second.Errors)
	}
	if first.Done != second.Done {
		t.Errorf("second pass did %d rows, first did %d", second.Done, first.Done)
	}
	if !slices.Equal(one, two) {
		t.Errorf("the second pass changed the vector again: %v -> %v", one, two)
	}
}

// The clause rows are the rows the offline tool cannot reach: they never enter
// doc_index.json, so a pass that walks the index skips them and long facts keep
// half-old vectors. They live in l3_clause and must be re-embedded like anything else.
func TestReembedCoversClauseRows(t *testing.T) {
	dir := t.TempDir()
	st := newReembedStore(t, dir, &oneHotEmbedder{pos: 0, dim: 8})
	// splitClauses cuts on ". ", "; " and newlines, so the fixture must use one of
	// those: a full-width "。" leaves the fact as one piece and writes no clause rows.
	long := strings.Repeat("This is a long memory sentence used to trigger clause indexing. ", 30)
	parent := mkFact(t, st, "alice", long)

	clauseCol := st.byName[clauseCollection]
	if clauseCol == nil {
		t.Fatal("no l3_clause collection")
	}
	if clauseCol.Count() == 0 {
		t.Fatal("the long fact produced no clause rows")
	}
	clauseID := clauseID(parent, 0)
	before := storedVector(t, st, clauseCollection, clauseID)

	st.embed = &oneHotEmbedder{pos: 6, dim: 8}
	rep := st.ReembedAll(context.Background(), allCollections(), 0, false)

	after := storedVector(t, st, clauseCollection, clauseID)
	if after[6] != 1 || after[0] != 0 {
		t.Errorf("clause vector was not re-embedded: %v -> %v", before, after)
	}
	if rep.Layers[clauseCollection] == 0 {
		t.Errorf("the pass reports no clause rows in its per-collection counts: %v", rep.Layers)
	}
}

// A limit bounds how many rows one pass touches, across collections.
func TestReembedHonoursLimit(t *testing.T) {
	dir := t.TempDir()
	st := newReembedStore(t, dir, &oneHotEmbedder{pos: 0, dim: 8})
	for i := 0; i < 5; i++ {
		mkFact(t, st, "alice", "fact number "+string(rune('a'+i)))
	}
	rep := st.ReembedAll(context.Background(), []string{string(memory.L3Fact)}, 2, true)
	if rep.Total != 2 || rep.Done != 2 {
		t.Errorf("pass with limit 2 touched %d rows (%d done)", rep.Total, rep.Done)
	}
}

// A model whose dimension changed must not be silently mixed into the collection: the
// pass reports it and leaves the row alone.
func TestReembedRefusesADimensionChange(t *testing.T) {
	dir := t.TempDir()
	st := newReembedStore(t, dir, &oneHotEmbedder{pos: 0, dim: 8})
	id := mkFact(t, st, "alice", "a fact")
	st.embed = &oneHotEmbedder{pos: 0, dim: 16} // the "new model" has another shape

	rep := st.ReembedAll(context.Background(), []string{string(memory.L3Fact)}, 0, false)
	if rep.Failed == 0 {
		t.Fatalf("a dimension change must be reported, got %+v", rep)
	}
	if !strings.Contains(strings.Join(rep.Errors, " "), "dimensions") {
		t.Errorf("the error must explain the dimension change: %v", rep.Errors)
	}
	if got := storedVector(t, st, string(memory.L3Fact), id); len(got) != 8 {
		t.Errorf("the row was overwritten with a %d-dimension vector", len(got))
	}
}

// The handler: the admin gate, POST-only, a dry run by default, a clear error for an
// unknown collection, and 409 while another pass holds the job.
func TestHandleReembedGatesAndDefaults(t *testing.T) {
	srv := newTestServer(t, "m", "http://127.0.0.1:1/v1")
	t.Cleanup(func() { srv.reembedBusy.Store(false) })

	call := func(body string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		srv.handleReembed(w, httptest.NewRequest("POST", "/api/v1/admin/reembed", strings.NewReader(body)))
		return w
	}

	// Admin off by default: the same gate as the other maintenance endpoints.
	if w := call(`{}`); w.Code != 403 {
		t.Errorf("admin off: status %d, want 403", w.Code)
	}
	srv.admin = true
	if w := call(`{"layers":["nope"]}`); w.Code != 400 {
		t.Errorf("unknown collection: status %d, want 400 (%s)", w.Code, w.Body.String())
	}
	if w := call(`{"dry_run":true}`); w.Code != 200 {
		t.Fatalf("empty dry run: status %d (%s)", w.Code, w.Body.String())
	} else {
		var out map[string]any
		if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
			t.Fatal(err)
		}
		if out["dry_run"] != true {
			t.Errorf("no body means a dry run, got %v", out["dry_run"])
		}
		if _, ok := out["errors"]; !ok {
			t.Errorf("the response must always carry errors[]: %s", w.Body.String())
		}
	}
}

// A second request while a pass is running is answered busy, not queued: the flag is
// taken before any embedding request is made.
func TestHandleReembedIsBusyForASecondRequest(t *testing.T) {
	srv := newTestServer(t, "m", "http://127.0.0.1:1/v1")
	srv.admin = true
	// Write the row with the server's own embedder, then swap in one that blocks, so
	// only the pass is held open (the write itself must not wait on the gate).
	if err := srv.store.Add(memory.L3Fact, newID(), "a fact to embed", map[string]string{"user_id": "u"}); err != nil {
		t.Fatal(err)
	}
	release := make(chan struct{})
	entered := make(chan struct{}, 1)
	srv.store.embed = &gateEmbedder{entered: entered, release: release}

	first := make(chan int, 1)
	go func() {
		w := httptest.NewRecorder()
		srv.handleReembed(w, httptest.NewRequest("POST", "/api/v1/admin/reembed",
			strings.NewReader(`{"dry_run":true}`)))
		first <- w.Code
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the first pass never reached the embedder")
	}
	w := httptest.NewRecorder()
	srv.handleReembed(w, httptest.NewRequest("POST", "/api/v1/admin/reembed", strings.NewReader(`{}`)))
	if w.Code != 409 {
		t.Errorf("a second pass: status %d, want 409 (%s)", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "busy") {
		t.Errorf("the busy answer must say so: %s", w.Body.String())
	}
	close(release)
	if code := <-first; code != 200 {
		t.Errorf("the first pass finished with %d, want 200", code)
	}
	// The flag must be free again, or every later pass would be refused.
	if srv.reembedBusy.Load() {
		t.Error("reembedBusy stayed set after the pass")
	}
}

// gateEmbedder blocks inside Embed until release is closed, so a test can hold a pass
// open at a known point.
type gateEmbedder struct {
	entered chan struct{}
	release chan struct{}
}

func (g *gateEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	select {
	case g.entered <- struct{}{}:
	default:
	}
	<-g.release
	return []float32{1, 0, 0}, nil
}

func (g *gateEmbedder) Dim() int { return 3 }

// newReembedStore opens a store over a temp dir with the given embedder.
func newReembedStore(t *testing.T, dir string, e Embedder) *MemoryStore {
	t.Helper()
	st, err := NewMemoryStore(ctxForTest(), dir, e, addGraphPath(dir), e.Dim())
	if err != nil {
		t.Skipf("MemoryStore unavailable in test env: %v", err)
	}
	t.Cleanup(st.Close)
	return st
}

// addGraphPath keeps the graph next to the store, as main does.
func addGraphPath(dir string) string { return dir + "/graph.json" }

// mkFact writes one L3 fact and returns its id.
func mkFact(t *testing.T, st *MemoryStore, user, text string) string {
	t.Helper()
	id := newID()
	if err := st.Add(memory.L3Fact, id, text, map[string]string{
		"user_id": user, "agent_id": "a", "ts": time.Now().UTC().Format(time.RFC3339)}); err != nil {
		t.Fatalf("add fact: %v", err)
	}
	return id
}
