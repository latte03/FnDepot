package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/tuancookiez-hub/hyatlas-v4/memory"
)

// Record only marks the counts. The file is written by Flush, or by the flush timer
// within usageFlushEvery, and a Flush with nothing new writes nothing.
func TestRecordDefersWrite(t *testing.T) {
	const text = "the extraction model is poolside and returns forbidden"
	dir := t.TempDir()
	path := filepath.Join(dir, usageLearningFile)
	u := NewUsageStats(dir)

	recordN(u, "m", text, "unrelated words", 1)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("file written inside Record: stat err = %v", err)
	}
	if err := u.Flush(); err != nil {
		t.Fatal(err)
	}
	if got := NewUsageStats(dir).m["m"].Shown; got != 1 {
		t.Fatalf("after Flush: shown = %d, want 1", got)
	}

	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := u.Flush(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("clean Flush rewrote the file: stat err = %v", err)
	}

	recordN(u, "m", text, "unrelated words", 1)
	deadline := time.Now().Add(usageFlushEvery + 3*time.Second)
	for {
		if _, err := os.Stat(path); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("flush timer did not write the counts")
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func TestUsedInReply(t *testing.T) {
	cases := []struct {
		name   string
		memory string
		reply  string
		query  string
		want   bool
	}{
		{"term overlap yes", "the sync worker batches writes every thirty seconds", "the sync worker batches all writes", "", true},
		{"term overlap no", "the sync worker batches writes every thirty seconds", "weather is mild and the sky is clear", "", false},
		{"identifier match", "sync server listens on port 19528", "I checked it, the server uses 19528 now", "", true},
		{"short memory without identifier", "likes coffee", "likes coffee a lot", "", false},
		{"empty memory", "", "anything", "", false},
		{"empty reply", "the sync worker batches writes", "", "", false},
		// The reply repeats the question; the memory added nothing it used.
		{"only question words", "HyAtlas v4.1.2 fixes extraction pipeline", "HyAtlas extraction now runs on stepfun", "what model does hyatlas use for extraction", false},
		{"beyond the question", "HyAtlas extraction switched from poolside to stepfun after 403s", "HyAtlas extraction runs on stepfun since poolside kept failing", "what model does hyatlas use for extraction", true},
		{"identifier the question named", "sync server listens on port 19528", "yes, 19528 is the port", "is 19528 the port", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := usedInReply(c.memory, c.reply, c.query); got != c.want {
				t.Fatalf("usedInReply(%q, %q) = %v, want %v", c.memory, c.reply, got, c.want)
			}
		})
	}
}

// cjkOverlap reports how many of the memory's distinct evidence terms come back in the
// reply, using the same admission rules as usedInReply. Tests log it so the ratio that
// decided the case is visible: a pass that only cleared the bar by a hair is worth
// knowing about when the next tokenisation change lands.
func cjkOverlap(memory, reply, query string) (hit, total int) {
	replyTerms := map[string]bool{}
	for _, t := range keywordTokens(reply) {
		replyTerms[t] = true
	}
	queryTerms := map[string]bool{}
	for _, t := range keywordTokens(query) {
		queryTerms[t] = true
	}
	seen := map[string]bool{}
	for _, t := range keywordTokens(memory) {
		if !isLongEnoughTerm(t) || seen[t] || queryTerms[t] {
			continue
		}
		seen[t] = true
		total++
		if replyTerms[t] {
			hit++
		}
	}
	return hit, total
}

// A memory written in Chinese is indexed as two-rune bigrams, so the old gate
// (four runes) dropped every one of its terms and usedInReply could only answer
// false — the usage feedback loop silently ignored every memory the server wrote
// after the language rule made Chinese the norm. The positive case pins that a
// Chinese reply really quoting the memory counts; the negative case pins that an
// equally long reply on another topic still does not.
func TestUsedInReplyChineseMemory(t *testing.T) {
	const memory = "混元记忆的快照周期默认是每六小时一次，可以在环境变量里调整；自动快照只保留最近七份，更早的会被清理掉。"
	const query = "docker 容器怎么挂载目录"

	cases := []struct {
		name  string
		reply string
		want  bool
	}{
		{
			name:  "quotes two clauses",
			reply: "快照周期默认是每六小时一次，自动快照只保留最近七份。",
			want:  true,
		},
		{
			name:  "same length other topic",
			reply: "他说的是对的，我们下次再一起讨论这个问题，顺便把文档更新一下。",
			want:  false,
		},
		{
			// The stress case for the 30% bar: a reply on another subject that still
			// repeats the generic pairs a Chinese sentence shares with any technical
			// text (可以/自动/调整/保留/默认/最近/清理). Bigrams make these cheap to
			// hit, so this is where a too-loose threshold would show up.
			name:  "generic pairs but other topic",
			reply: "系统可以自动调整参数，我们保留原来的默认值，最近没有再清理缓存。",
			want:  false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			hit, total := cjkOverlap(memory, c.reply, query)
			t.Logf("evidence terms: %d of %d (%.0f%%)", hit, total, 100*float64(hit)/float64(total))
			if total == 0 {
				t.Fatal("no evidence terms admitted: the gate dropped every Chinese bigram")
			}
			if got := usedInReply(memory, c.reply, query); got != c.want {
				t.Fatalf("usedInReply = %v, want %v (overlap %d/%d)", got, c.want, hit, total)
			}
		})
	}
}

// The English path must not move: a Latin token still needs four runes to count, so
// every token the new predicate admits is one the old rule admitted too.
func TestUsedInReplyEnglishGateUnchanged(t *testing.T) {
	const memory = "the snapshot interval lives in HYATLAS_SNAPSHOT_EVERY and defaults to six hours"
	for _, tok := range keywordTokens(memory) {
		if isLongEnoughTerm(tok) && utf8.RuneCountInString(tok) < 4 {
			t.Fatalf("token %q admitted by the new gate but rejected by the old one", tok)
		}
	}
	if !usedInReply(memory, "the snapshot interval defaults to six hours", "") {
		t.Fatal("English overlap no longer counts as used")
	}
	if usedInReply(memory, "weather is mild and the sky is clear", "") {
		t.Fatal("unrelated English reply counted as used")
	}
}

// recordN records n recalls of one memory, each with the given reply.
func recordN(u *UsageStats, id, text, reply string, n int) {
	for i := 0; i < n; i++ {
		u.Record([]string{id}, map[string]string{id: text}, reply, "")
	}
}

func TestRecordAndFactor(t *testing.T) {
	const text = "the extraction model is poolside and returns forbidden"
	dir := t.TempDir()
	u := NewUsageStats(dir)

	recordN(u, "few", text, "unrelated words", 2)
	if f := u.Factor("few"); f != 1.0 {
		t.Fatalf("below 3 shows: factor = %v, want 1.0", f)
	}

	recordN(u, "ignored", text, "unrelated words", 10)
	if f := u.Factor("ignored"); f >= 1.0 {
		t.Fatalf("10 shows, 0 used: factor = %v, want < 1.0", f)
	}

	recordN(u, "used", text, text, 10)
	if f := u.Factor("used"); f <= 1.0 {
		t.Fatalf("10 shows, 10 used: factor = %v, want > 1.0", f)
	}
	if f := u.Factor("unknown"); f != 1.0 {
		t.Fatalf("unknown id: factor = %v, want 1.0", f)
	}

	if err := u.Flush(); err != nil {
		t.Fatal(err)
	}
	reloaded := NewUsageStats(dir)
	if a, b := reloaded.Factor("used"), u.Factor("used"); a != b {
		t.Fatalf("reloaded factor = %v, want %v", a, b)
	}
	if a, b := reloaded.Factor("ignored"), u.Factor("ignored"); a != b {
		t.Fatalf("reloaded factor for ignored = %v, want %v", a, b)
	}
}

func TestRerankUsage(t *testing.T) {
	const text = "the extraction model is poolside and returns forbidden"
	u := NewUsageStats(t.TempDir())
	recordN(u, "ignored", text, "unrelated words", 10)
	recordN(u, "used", text, text, 10)

	hits := []SearchHit{{ID: "ignored", Score: 0.5}, {ID: "used", Score: 0.5}}
	out := rerankUsage(hits, u)
	if len(out) != 2 || out[0].ID != "used" || out[1].ID != "ignored" {
		t.Fatalf("rerank order = %v, want used before ignored", out)
	}
	if out[0].Score != 0.5 || out[1].Score != 0.5 {
		t.Fatalf("rerank changed Score: %v %v", out[0].Score, out[1].Score)
	}
	if hits[0].ID != "ignored" {
		t.Fatalf("rerank mutated the input slice")
	}

	same := rerankUsage(hits, nil)
	if len(same) != 2 || same[0].ID != "ignored" || same[1].ID != "used" {
		t.Fatalf("nil stats changed order: %v", same)
	}
}

func TestHandleFeedback(t *testing.T) {
	dir := t.TempDir()
	store, err := NewMemoryStore(context.Background(), dir, NewLocalEmbedder(384), filepath.Join(dir, "graph.json"), 384)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(store.Close)

	const uText = "the sync worker batches writes every thirty seconds"
	meta := func(user string) map[string]string {
		return map[string]string{"user_id": user, "agent_id": "a"}
	}
	if err := store.Add(memory.L3Fact, "u1", uText, meta("u")); err != nil {
		t.Fatal(err)
	}
	if err := store.Add(memory.L3Fact, "u2", "cache entries expire after an hour", meta("u")); err != nil {
		t.Fatal(err)
	}
	if err := store.Add(memory.L3Fact, "x1", "billing runs on the first of the month", meta("x")); err != nil {
		t.Fatal(err)
	}

	u := NewUsageStats(dir)
	h := u.HandleFeedback(store, nil)
	body, _ := json.Marshal(map[string]any{
		"recalled_ids": []string{"u1", "u2", "x1"},
		"reply":        "the sync worker batches all writes every thirty seconds",
		"user_id":      "u",
		"agent_id":     "a",
	})
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/api/v1/feedback", bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body %s", rec.Code, rec.Body.String())
	}
	var got struct {
		OK    bool `json:"ok"`
		Shown int  `json:"shown"`
		Used  int  `json:"used"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.OK || got.Shown != 2 || got.Used != 1 {
		t.Fatalf("response = %+v, want ok shown=2 used=1", got)
	}
	if e := u.m["x1"]; e.Shown != 0 {
		t.Fatalf("other owner's memory was counted: %+v", e)
	}

	bad := httptest.NewRecorder()
	h(bad, httptest.NewRequest(http.MethodPost, "/api/v1/feedback", bytes.NewReader([]byte("{not json"))))
	if bad.Code != http.StatusBadRequest {
		t.Fatalf("bad body status = %d, want 400", bad.Code)
	}
}

// A memory written under an alias of the caller's user ID counts; with no alias it
// is another owner's and does not.
func TestHandleFeedbackAliases(t *testing.T) {
	dir := t.TempDir()
	store, err := NewMemoryStore(context.Background(), dir, NewLocalEmbedder(384), filepath.Join(dir, "graph.json"), 384)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(store.Close)
	if err := store.Add(memory.L3Fact, "d1", "the sync worker batches writes every thirty seconds",
		map[string]string{"user_id": "discord-123", "agent_id": "a"}); err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(map[string]any{"recalled_ids": []string{"d1"},
		"reply": "the sync worker batches all writes every thirty seconds", "user_id": "default"})
	post := func(expand func([]string) []string) int {
		u := NewUsageStats(t.TempDir())
		rec := httptest.NewRecorder()
		u.HandleFeedback(store, expand)(rec, httptest.NewRequest(http.MethodPost, "/api/v1/feedback", bytes.NewReader(body)))
		var got struct{ Shown int }
		_ = json.Unmarshal(rec.Body.Bytes(), &got)
		return got.Shown
	}
	if n := post(nil); n != 0 {
		t.Fatalf("without aliases shown = %d, want 0", n)
	}
	alias := func(ids []string) []string { return append(ids, "discord-123") }
	if n := post(alias); n != 1 {
		t.Fatalf("with aliases shown = %d, want 1", n)
	}
}

// Past usageMaxEntries the map is pruned to 90% of the cap: never-used memories go
// first, so a heavily used one survives and an unused one does not.
func TestUsagePrunes(t *testing.T) {
	dir := t.TempDir()
	u := NewUsageStats(dir)
	for i := 0; i < usageMaxEntries+1; i++ {
		u.m[fmt.Sprintf("id-%d", i)] = usageEntry{Shown: 1}
	}
	u.m["heavy"] = usageEntry{Shown: 50, Used: 40, Last: "2026-01-01T00:00:00Z"}

	u.Record([]string{"probe"}, map[string]string{"probe": "unrelated"}, "reply", "")

	if got, want := len(u.m), usageMaxEntries*9/10; got != want {
		t.Fatalf("size after prune = %d, want %d", got, want)
	}
	if _, ok := u.m["heavy"]; !ok {
		t.Fatal("heavily used entry was pruned")
	}
	if _, ok := u.m["id-0"]; ok {
		t.Fatal("never-used entry id-0 survived the prune")
	}
}
