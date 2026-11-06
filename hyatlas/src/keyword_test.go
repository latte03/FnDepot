package main

import (
	"math"
	"reflect"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"

	"github.com/tuancookiez-hub/hyatlas-v4/memory"
)

// TestKeywordTokens checks that identifiers split on punctuation and stopwords drop out.
func TestKeywordTokens(t *testing.T) {
	got := keywordTokens("What is HYATLAS_SYNC_EXTRACT on port 19528? RSI(2) Reg-T")
	want := []string{"hyatla", "sync", "extract", "port", "19528", "rsi", "reg"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("keywordTokens: want %v, got %v", want, got)
	}
}

// TestKeywordTokensChineseBigrams checks that a Chinese run is indexed as overlapping
// character bigrams. As one token per clause, a Chinese memory could only match a query
// that repeated the clause character for character.
func TestKeywordTokensChineseBigrams(t *testing.T) {
	got := keywordTokens("混元记忆的快照周期")
	want := []string{"混元", "元记", "记忆", "忆的", "的快", "快照", "照周", "周期"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("keywordTokens: want %v, got %v", want, got)
	}
}

// TestKeywordTokensMixedScript checks a run that holds both scripts: the Latin words and
// the Chinese bigrams of its parts, and no whole-clause token.
func TestKeywordTokensMixedScript(t *testing.T) {
	got := keywordTokens("Docker容器没加pid=host会有进程回收问题")
	have := make(map[string]bool, len(got))
	for _, tok := range got {
		have[tok] = true
	}
	for _, want := range []string{"docker", "容器", "器没", "回收", "收问", "问题", "host"} {
		if !have[want] {
			t.Errorf("keywordTokens is missing %q; got %v", want, got)
		}
	}
	for _, tok := range got {
		if isCJKSegment(tok) && utf8.RuneCountInString(tok) >= 3 {
			t.Errorf("keywordTokens emitted a whole-clause token %q; got %v", tok, got)
		}
	}
}

// TestKeywordTokensSingleHanIsDropped checks that one character alone indexes nothing,
// the same gate as a one-character Latin run: with no neighbour it would match every
// memory holding that character.
func TestKeywordTokensSingleHanIsDropped(t *testing.T) {
	if got := keywordTokens("看 改 好"); len(got) != 0 {
		t.Fatalf("want no tokens, got %v", got)
	}
}

// TestKeywordRequired checks how many query terms a document must match.
func TestKeywordRequired(t *testing.T) {
	cases := map[int]int{1: 1, 3: 3, 4: 3, 5: 4, 8: 6}
	for n, want := range cases {
		if got := keywordRequired(n); got != want {
			t.Errorf("keywordRequired(%d): want %d, got %d", n, want, got)
		}
	}
}

// TestBM25RankNeedsAllTermsOnShortQuery checks that a short query needs every term.
func TestBM25RankNeedsAllTermsOnShortQuery(t *testing.T) {
	docs := []DocIndex{
		{ID: "d1", Content: "HYATLAS_SYNC_EXTRACT=on makes the write block"},
		{ID: "d2", Content: "sync your watch daily"},
		{ID: "d3", Content: "nothing relevant"},
	}
	got := bm25Rank(keywordTokens("HYATLAS_SYNC_EXTRACT"), docs)
	if len(got) != 1 {
		t.Fatalf("want 1 result, got %d", len(got))
	}
	if got[0].Doc.ID != "d1" {
		t.Errorf("want ID d1, got %s", got[0].Doc.ID)
	}
	if got[0].Matched != 3 {
		t.Errorf("want Matched 3, got %d", got[0].Matched)
	}
}

// TestBM25RankPrefersShorterDoc checks that the shorter of two equal matches ranks first.
func TestBM25RankPrefersShorterDoc(t *testing.T) {
	docs := []DocIndex{
		{ID: "d2", Content: "nvda dip strategy plus a long tail of unrelated filler words about many other separate topics entirely"},
		{ID: "d1", Content: "nvda dip strategy"},
	}
	got := bm25Rank(keywordTokens("NVDA dip"), docs)
	if len(got) != 2 {
		t.Fatalf("want 2 results, got %d", len(got))
	}
	if got[0].Doc.ID != "d1" {
		t.Errorf("want d1 first, got %s", got[0].Doc.ID)
	}
}

// TestCosineSim checks cosine similarity on identical, orthogonal, empty and mismatched vectors.
func TestCosineSim(t *testing.T) {
	if got := cosineSim([]float32{1, 2, 3}, []float32{1, 2, 3}); math.Abs(got-1) > 1e-6 {
		t.Errorf("identical: want 1, got %v", got)
	}
	if got := cosineSim([]float32{1, 0}, []float32{0, 1}); math.Abs(got) > 1e-6 {
		t.Errorf("orthogonal: want 0, got %v", got)
	}
	if got := cosineSim(nil, []float32{1}); got != 0 {
		t.Errorf("empty: want 0, got %v", got)
	}
	if got := cosineSim([]float32{1, 2}, []float32{1, 2, 3}); got != 0 {
		t.Errorf("length mismatch: want 0, got %v", got)
	}
}

// TestKeywordSearchScopesAndSkipsRawAndSuperseded checks owner, agent, layer and
// superseded filtering on a real store.
func TestKeywordSearchScopesAndSkipsRawAndSuperseded(t *testing.T) {
	srv := newTestServer(t, "m", "http://127.0.0.1:1/v1")
	a, b, c, d, e := newID(), newID(), newID(), newID(), newID()
	mustAdd := func(layer memory.Layer, id, text, user string) {
		t.Helper()
		meta := map[string]string{"user_id": user, "agent_id": "a1"}
		if err := srv.store.Add(layer, id, text, meta); err != nil {
			t.Fatal(err)
		}
	}
	mustAdd(memory.L3Fact, a, "HYATLAS_SYNC_EXTRACT=on makes the write block", "alice")
	mustAdd(memory.L3Fact, b, "the sync button is blue", "alice")
	mustAdd(memory.L3Fact, c, "HYATLAS_SYNC_EXTRACT is a server knob", "bob")
	mustAdd(memory.L2Raw, d, "raw turn mentioning HYATLAS_SYNC_EXTRACT", "alice")
	mustAdd(memory.L3Fact, e, "old note: HYATLAS_SYNC_EXTRACT defaulted to on", "alice")
	if _, err := srv.store.Supersede([]string{e}, a); err != nil {
		t.Fatal(err)
	}

	hits, err := srv.store.KeywordSearch("HYATLAS_SYNC_EXTRACT", 5, "", []string{"alice"}, "a1")
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("alice scoped: want 1 hit, got %d", len(hits))
	}
	if hits[0].ID != a {
		t.Errorf("alice scoped: want ID %s, got %s", a, hits[0].ID)
	}
	if hits[0].Layer != memory.L3Fact {
		t.Errorf("layer: want %s, got %s", memory.L3Fact, hits[0].Layer)
	}
	if hits[0].Meta["user_id"] != "alice" {
		t.Errorf("meta user_id: want alice, got %q", hits[0].Meta["user_id"])
	}
	if hits[0].Score <= 0 {
		t.Errorf("score: want > 0, got %v", hits[0].Score)
	}

	raw, err := srv.store.KeywordSearch("HYATLAS_SYNC_EXTRACT", 5, memory.L2Raw, []string{"alice"}, "a1")
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 1 || raw[0].ID != d {
		t.Errorf("L2Raw: want exactly hit %s, got %d hits", d, len(raw))
	}

	all, err := srv.store.KeywordSearch("HYATLAS_SYNC_EXTRACT", 5, "", nil, "a1")
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 2 {
		t.Fatalf("no owner filter: want 2 hits, got %d", len(all))
	}
	ids := map[string]bool{all[0].ID: true, all[1].ID: true}
	if !ids[a] || !ids[c] {
		t.Errorf("no owner filter: want hits %s and %s, got %v", a, c, ids)
	}
}

func TestSameNumbers(t *testing.T) {
	cases := []struct {
		a, b string
		want bool
	}{
		{"open PR #101722", "open PR #96783", false},
		{"Tuna prefers Vue 3 Composition API", "Tuna prefers Vue 3 with Composition API", true},
		{"runs on 127.0.0.1:19528", "server at 127.0.0.1:19528", true},
		{"version 4.4.0", "version 4.5.0", false},
		{"no numbers here", "none here either", true},
		{"port 8080", "the port", false},
	}
	for _, c := range cases {
		if got := sameNumbers(c.a, c.b); got != c.want {
			t.Errorf("sameNumbers(%q, %q) = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}

// kwFixtureDocs is a small corpus shaped like the live store: a note that states the
// embed-limit numbers without quoting any phrase, a model-selection note, two snapshot
// notes that word the same fact differently, a row that shares one bigram, and a row
// that shares nothing.
func kwFixtureDocs() []DocIndex {
	return []DocIndex{
		{ID: "embed-limit", Content: "上游对 embed 输入的限制是 929 个字符,超出会拿到错误码 400 和 20015"},
		{ID: "bge-select", Content: "选型结论是 SiliconFlow 上的 BAAI/bge-m3 更适合中文长文本"},
		{ID: "snapshot", Content: "快照周期由 HYATLAS_SNAPSHOT_EVERY 控制,默认每六小时"},
		{ID: "snapshot-edit", Content: "改快照周期要编辑 /vol1/@appdata/hyatlas/env 然后重启"},
		{ID: "snapshot-reword", Content: "快照周期怎么调整都在 env 里改"},
		{ID: "one-bigram", Content: "今天做了一次快照备份"},
		{ID: "unrelated", Content: "NAS 的 IP 地址是 192.168.0.51"},
	}
}

// legacyKeywordTokens is keywordTokens as it was before Chinese was indexed in bigrams:
// one Chinese run is one token. It exists so a test can prove the gate never got
// stricter than the rule it replaced.
func legacyKeywordTokens(text string) []string {
	fields := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if utf8.RuneCountInString(f) < 2 || keywordStopwords[f] {
			continue
		}
		out = append(out, stemTerm(f))
	}
	return out
}

// gateHits is the flat-token gate: how many distinct query terms a document contains,
// and whether that reaches keywordRequired. It is the rule the gate used before units.
func gateHits(tokenise func(string) []string, query string, doc DocIndex) (int, bool) {
	seen := make(map[string]bool)
	var terms []string
	for _, t := range tokenise(query) {
		if !seen[t] {
			seen[t] = true
			terms = append(terms, t)
		}
	}
	have := make(map[string]bool)
	for _, t := range tokenise(doc.Content) {
		have[t] = true
	}
	hits := 0
	for _, t := range terms {
		if have[t] {
			hits++
		}
	}
	return hits, hits >= keywordRequired(len(terms))
}

// gateAdmits is gateHits over a corpus, keyed by document id.
func gateAdmits(tokenise func(string) []string, query string, docs []DocIndex) map[string]bool {
	out := make(map[string]bool, len(docs))
	for _, d := range docs {
		if _, ok := gateHits(tokenise, query, d); ok {
			out[d.ID] = true
		}
	}
	return out
}

// rankHits is the shipped gate over a corpus: id to the number of query units it hit.
func rankHits(query string, docs []DocIndex) map[string]int {
	out := make(map[string]int, len(docs))
	for _, m := range bm25Rank(keywordTokens(query), docs) {
		out[m.Doc.ID] = m.Matched
	}
	return out
}

// mustDoc returns the fixture with id.
func mustDoc(t *testing.T, docs []DocIndex, id string) DocIndex {
	t.Helper()
	for _, d := range docs {
		if d.ID == id {
			return d
		}
	}
	t.Fatalf("fixture is missing document %q", id)
	return DocIndex{}
}

// TestQueryUnitsGroupBigramsIntoPhrases checks that a run of overlapping Chinese bigrams
// comes back as the phrase the user typed, and that everything else stays a unit of its
// own.
func TestQueryUnitsGroupBigramsIntoPhrases(t *testing.T) {
	cases := []struct {
		query string
		want  []int
	}{
		{"混元记忆的快照周期，怎么改", []int{8, 2}},
		{"929 字以上会返回 400/20015", []int{1, 5, 1, 1}},
	}
	for _, c := range cases {
		units := groupQueryUnits(keywordTokens(c.query))
		sizes := make([]int, 0, len(units))
		for _, u := range units {
			sizes = append(sizes, len(u))
		}
		if !reflect.DeepEqual(sizes, c.want) {
			t.Errorf("groupQueryUnits(%q) sizes: want %v, got %v (%q)", c.query, c.want, sizes, units)
		}
	}
}

// TestQueryUnitsKeepChainsAcrossRepeats checks that a bigram repeated earlier in the
// query does not cut the phrase it also sits in. Grouping the deduplicated term list did
// exactly that — the second 输出 was gone, so 该输 and 出判 stopped overlapping and
// "因此不能只凭该输出判定失败" arrived as two units instead of one, which left a real
// memory one unit short of the gate.
func TestQueryUnitsKeepChainsAcrossRepeats(t *testing.T) {
	units := groupQueryUnits(keywordTokens("输出 因此不能只凭该输出判定失败"))
	if len(units) != 2 {
		t.Fatalf("want 2 units, got %d: %q", len(units), units)
	}
	if len(units[0]) != 1 || len(units[1]) != 12 {
		t.Errorf("want sizes [1 12], got [%d %d]: %q", len(units[0]), len(units[1]), units)
	}
}

// TestKeywordGateCountsPhrasesNotBigrams pins the regression that shipped in 5.0.0-5: a
// query mixing numbers and a Chinese clause was broken into eight flat bigram terms, so
// the note that states the numbers without quoting the clause fell to 3 hits against a
// threshold of 6 and disappeared from results. It has to come back at the count the
// pre-bigram gate gave it.
func TestKeywordGateCountsPhrasesNotBigrams(t *testing.T) {
	docs := kwFixtureDocs()
	query := "929 字以上会返回 400/20015"
	limit := mustDoc(t, docs, "embed-limit")

	legacyHits, legacyOK := gateHits(legacyKeywordTokens, query, limit)
	if !legacyOK || legacyHits < 3 {
		t.Fatalf("fixture no longer covers the old rule: legacy hits %d, admits %v", legacyHits, legacyOK)
	}
	flatHits, flatOK := gateHits(keywordTokens, query, limit)
	if flatOK {
		t.Fatalf("fixture no longer reproduces the regression: flat bigram gate admits it with %d hits", flatHits)
	}

	got, ok := rankHits(query, docs)["embed-limit"]
	if !ok {
		t.Fatalf("the embed-limit note is gone: flat gate counted %d hits, the units rule dropped it anyway", flatHits)
	}
	if got < legacyHits {
		t.Errorf("matched %d units, want at least the %d hits the pre-bigram gate counted", got, legacyHits)
	}
}

// TestKeywordGateKeepsF5bGains checks that a reworded Chinese note — one that shares
// enough bigrams to be recognisably about the same thing but never quotes the phrase —
// stays in the results, and that the gate is no stricter than the flat bigram gate.
func TestKeywordGateKeepsF5bGains(t *testing.T) {
	docs := kwFixtureDocs()
	query := "快照周期怎么改"

	for _, id := range []string{"snapshot-edit", "snapshot-reword"} {
		if _, ok := rankHits(query, docs)[id]; !ok {
			t.Errorf("%s dropped: a note that shares most of the phrase is a match", id)
		}
	}
	if _, ok := gateHits(legacyKeywordTokens, query, mustDoc(t, docs, "snapshot-reword")); ok {
		t.Errorf("fixture no longer covers the gain: the pre-bigram gate already admitted it")
	}
	flat := gateAdmits(keywordTokens, query, docs)
	got := rankHits(query, docs)
	for id := range flat {
		if _, ok := got[id]; !ok {
			t.Errorf("%s dropped: the flat bigram gate admitted it and units must not be stricter", id)
		}
	}
}

// TestKeywordGateRejectsUnrelatedDoc checks that sharing one bigram, or nothing, still
// fails: a loose gate that returns an unrelated memory is worse than a missing one.
func TestKeywordGateRejectsUnrelatedDoc(t *testing.T) {
	docs := kwFixtureDocs()
	got := rankHits("快照周期怎么改", docs)
	for _, id := range []string{"unrelated", "one-bigram", "embed-limit", "bge-select"} {
		if n, ok := got[id]; ok {
			t.Errorf("%s returned with %d units hit: it has nothing to do with the query", id, n)
		}
	}
}

// TestKeywordGateNeverGetsStricter is the property the fix has to keep: every document
// the pre-bigram gate admitted is still admitted. A Chinese run was one token then, and
// a document that contained it contains all of its bigrams, so a phrase unit is always
// hit — the unit count can only differ by being looser.
func TestKeywordGateNeverGetsStricter(t *testing.T) {
	docs := kwFixtureDocs()
	queries := []string{
		"929 字以上会返回 400/20015",
		"快照周期怎么改",
		"混元记忆的快照周期，怎么改",
		"HYATLAS_SNAPSHOT_EVERY 怎么改",
		"bge-m3 和 bge-large 选哪个",
		"快照周期",
	}
	for _, q := range queries {
		got := rankHits(q, docs)
		for id := range gateAdmits(legacyKeywordTokens, q, docs) {
			if _, ok := got[id]; !ok {
				t.Errorf("query %q: %s was admitted before Chinese was indexed in bigrams and is now dropped", q, id)
			}
		}
	}
}
