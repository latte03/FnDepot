package main

import (
	"encoding/json"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/tuancookiez-hub/hyatlas-v4/memory"
)

// searchHitIDs runs one search through the handler and returns the ids of every hit
// in every channel.
func searchHitIDs(t *testing.T, srv *Server, query string, minScore float64) []string {
	t.Helper()
	body := `{"query":"` + query + `","user_id":"u","agent_id":"a","limit":5,"min_score":` +
		strconv.FormatFloat(minScore, 'f', 2, 64) + `}`
	w := httptest.NewRecorder()
	srv.handleSearch(w, httptest.NewRequest("POST", "/api/v1/search", strings.NewReader(body)))
	if w.Code != 200 {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var resp struct {
		Memories map[string][]map[string]any `json:"memories"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	var ids []string
	for _, ch := range resp.Memories {
		for _, h := range ch {
			if id, ok := h["memory_id"].(string); ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

// hasAnyID reports whether ids holds want.
func hasAnyID(ids []string, want string) bool {
	for _, id := range ids {
		if id == want {
			return true
		}
	}
	return false
}

// A natural-language question is not an identifier query, whatever punctuation it
// carries. An in-sentence separator used to make the whole question "identifier-like"
// (the separator sat between two letters), which sent it through requireTerms: every
// hit had to contain both halves as literal tokens, so the answer was empty.
func TestIdentifierTermsRejectsSentencePunctuation(t *testing.T) {
	const base = "混元记忆的快照周期怎么改"
	for _, sep := range []string{"，", "、", ",", "。", "；", "：", "？"} {
		q := strings.Replace(base, "快照周期", "快照周期"+sep, 1)
		if got := identifierTerms(q); len(got) != 0 {
			t.Errorf("identifierTerms(%q) = %v, want none: a sentence, not an identifier", q, got)
		}
	}
	// '=' joins two words of a sentence the same way: the query is prose.
	if got := identifierTerms("Docker容器没加pid=host会有进程回收问题"); len(got) != 0 {
		t.Errorf("identifierTerms(pid=host sentence) = %v, want none", got)
	}
	// A real identifier still counts, so the precision filter keeps working.
	for _, c := range []struct{ q, want string }{
		{"pid=host", "pid,host"},
		{"HYATLAS_SYNC_EXTRACT", "hyatla,sync,extract"},
		{"RSI(2)", "rsi"},
		{"19528", "19528"},
		{"discord-tps plugin", "discord,tps"},
	} {
		if got := strings.Join(identifierTerms(c.q), ","); got != c.want {
			t.Errorf("identifierTerms(%q) = %q, want %q", c.q, got, c.want)
		}
	}
}

// A separator splits one question into two topics. Both memories must come back: the
// result is the union of what each clause finds, not an empty intersection.
func TestSearchSeparatorReturnsBothClauses(t *testing.T) {
	const query = "alpha，beta"
	const textA = "alpha snapshot interval is six hours"
	const textB = "beta change procedure lives in the panel"
	srv := newTestServer(t, "m", "http://127.0.0.1:1/v1")
	// One query vector, two memories on opposite sides of it: both clear the floor and
	// stay below the dedupe threshold, so neither one absorbs the other.
	srv.store = guardStore(t, vecEmbedder{
		query: {1, 0, 0},
		textA: {0.9, 0.43589, 0},
		textB: {0.9, -0.43589, 0},
	})
	a := addRow(t, srv.store, memory.L3Fact, textA, nowTS())
	b := addRow(t, srv.store, memory.L3Fact, textB, nowTS())

	ids := searchHitIDs(t, srv, query, 0.5)
	if !hasAnyID(ids, a) || !hasAnyID(ids, b) {
		t.Fatalf("separator query returned %v, want both %s and %s", ids, a, b)
	}
}

// The '=' case: a sentence that carries an equals sign is still a sentence. It used to
// be read as an identifier query, and requireTerms then dropped every hit that did not
// contain the two glued halves of the question.
func TestSearchEqualsInSentenceIsNotAnIdentifier(t *testing.T) {
	const query = "Docker容器没加pid=host会有进程回收问题"
	const text = "Docker 容器不共享主机 PID 命名空间时，网关记录的进程号会过期"
	srv := newTestServer(t, "m", "http://127.0.0.1:1/v1")
	srv.store = guardStore(t, vecEmbedder{query: {1, 0, 0}, text: simTo(0.9)})
	id := addRow(t, srv.store, memory.L3Fact, text, nowTS())

	if ids := searchHitIDs(t, srv, query, 0.5); !hasAnyID(ids, id) {
		t.Fatalf("'=' sentence returned %v, want %s", ids, id)
	}
}

// The identifier filter narrows the fused ranking; it never empties it. When no memory
// mentions the identifier, the fused hits stand instead of the whole query going to 0.
func TestIdentifierFilterNeverEmptiesTheQuery(t *testing.T) {
	const query = "HYATLAS_SYNC_EXTRACT"
	const text = "the panel restarts the service after a config change"
	srv := newTestServer(t, "m", "http://127.0.0.1:1/v1")
	srv.store = guardStore(t, vecEmbedder{query: {1, 0, 0}, text: simTo(0.9)})
	id := addRow(t, srv.store, memory.L3Fact, text, nowTS())

	if ids := searchHitIDs(t, srv, query, 0.5); !hasAnyID(ids, id) {
		t.Fatalf("identifier query with no mention returned %v, want the fused hit %s", ids, id)
	}
}
