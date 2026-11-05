package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/tuancookiez-hub/hyatlas-v4/memory"
)

// recordingEmbedder delegates to a function and remembers every text it was asked to
// embed, so a test can assert what the decorator actually sent. accepted keeps only
// the texts that were taken, because a piece the model rejected is expected.
type recordingEmbedder struct {
	fn       func(ctx context.Context, text string) ([]float32, error)
	texts    []string
	accepted []string
	dim      int
}

func (r *recordingEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	r.texts = append(r.texts, text)
	v, err := r.fn(ctx, text)
	if err == nil {
		r.accepted = append(r.accepted, text)
	}
	return v, err
}

func (r *recordingEmbedder) Dim() int { return r.dim }

func (r *recordingEmbedder) ModelName() string { return "recording" }

// maxAccepted returns the longest text the model took, in runes.
func (r *recordingEmbedder) maxAccepted() int {
	m := 0
	for _, t := range r.accepted {
		if n := utf8.RuneCountInString(t); n > m {
			m = n
		}
	}
	return m
}

// additiveEmbedder is linear: its vector is the sum of one weight per rune, with no
// normalization. That is what makes mean pooling checkable — the mean of the chunk
// vectors of a text is the whole text's vector divided by the chunk count, so
// re-normalizing it must give exactly the whole text's normalized vector.
type additiveEmbedder struct{ dim int }

func (a additiveEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	v := make([]float32, a.dim)
	for i, r := range []rune(text) {
		v[i%a.dim] += float32(r%7 + 1)
	}
	return v, nil
}

func (a additiveEmbedder) Dim() int          { return a.dim }
func (a additiveEmbedder) ModelName() string { return "additive" }

// lengthLimitedEmbedder mimics the remote model: it answers HTTP 400 (SiliconFlow's
// code 20015 shape) for any text over limit runes.
type lengthLimitedEmbedder struct {
	limit int
	dim   int
	inner Embedder
	calls int
	seen  []string
	// accepted holds only the texts the model took: a rejected 390-rune attempt is
	// exactly what the decorator is supposed to try and then split.
	accepted []string
}

func (l *lengthLimitedEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	l.calls++
	l.seen = append(l.seen, text)
	if utf8.RuneCountInString(text) > l.limit {
		return nil, fmt.Errorf("embed HTTP 400 (model bge-large-zh-v1.5, %d runes): "+
			`{"code":20015,"message":"Request body is too large"}`,
			utf8.RuneCountInString(text))
	}
	l.accepted = append(l.accepted, text)
	return l.inner.Embed(ctx, text)
}

func (l *lengthLimitedEmbedder) Dim() int          { return l.dim }
func (l *lengthLimitedEmbedder) ModelName() string { return "bge-large-zh-v1.5" }

// A text inside the budget is passed through untouched: one request, same vector, no
// pooling rounding.
func TestChunkedEmbedderPassesShortTextThrough(t *testing.T) {
	rec := &recordingEmbedder{dim: 4, fn: func(_ context.Context, s string) ([]float32, error) {
		return []float32{1, 0, 0, 0}, nil
	}}
	e := NewChunkedEmbedder(rec, 400)
	got, err := e.Embed(context.Background(), "短文本")
	if err != nil {
		t.Fatal(err)
	}
	if len(rec.texts) != 1 || rec.texts[0] != "短文本" {
		t.Fatalf("inner saw %q, want the text once and unchanged", rec.texts)
	}
	if len(got) != 4 || got[0] != 1 {
		t.Fatalf("vector = %v, want the inner vector", got)
	}
}

// Mean pooling is algebraically consistent: with a linear embedder, pooling the
// chunks of a text must give the same unit vector as embedding the text whole, even
// though the requests are smaller.
func TestChunkedEmbedderMeanPoolsLikeTheWholeText(t *testing.T) {
	inner := additiveEmbedder{dim: 8}
	// Three 16-rune sentences: 48 runes, 20-rune budget, so exactly 3 pieces.
	sentence := strings.Repeat("测", 15) + "。"
	text := sentence + sentence + sentence
	chunks := splitEmbedChunks(text, 20)
	if len(chunks) != 3 {
		t.Fatalf("splitter produced %d chunks, want 3: %q", len(chunks), chunks)
	}
	if got := strings.Join(chunks, ""); got != text {
		t.Fatalf("chunks do not concatenate back to the text: %q", got)
	}

	e := NewChunkedEmbedder(inner, 20)
	got, err := e.Embed(context.Background(), text)
	if err != nil {
		t.Fatal(err)
	}
	whole, err := inner.Embed(context.Background(), text)
	if err != nil {
		t.Fatal(err)
	}
	want := normalize(whole)

	if len(got) != len(want) {
		t.Fatalf("pooled vector has %d dims, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("pooled vector %v != whole-text vector %v (element %d)", got, want, i)
		}
	}
	var norm float64
	for _, x := range got {
		norm += float64(x) * float64(x)
	}
	if math.Abs(math.Sqrt(norm)-1) > 1e-5 {
		t.Errorf("pooled vector is not normalized: |v| = %v", math.Sqrt(norm))
	}
}

// A text that fits the rune budget but is still rejected must be split, not failed:
// the model's limit is in tokens, so no rune budget is safe for every content. The
// model here refuses anything over 80 runes while the budget is 400 — the live shape
// of "510 runes passed, 511 did not".
func TestChunkedEmbedderSplitsWhenShortTextIsRejected(t *testing.T) {
	rec := &recordingEmbedder{dim: 8, fn: func(_ context.Context, s string) ([]float32, error) {
		if utf8.RuneCountInString(s) > 80 {
			return nil, fmt.Errorf("embed HTTP 400 (model m, %d runes): %s",
				utf8.RuneCountInString(s), `{"code":20015,"message":"input is too long"}`)
		}
		return []float32{1, 0, 0, 0, 0, 0, 0, 0}, nil
	}}
	// 300 runes, well inside the 400 budget, but the model refuses anything over 80.
	text := strings.Repeat("内容", 150)
	if n := utf8.RuneCountInString(text); n != 300 {
		t.Fatalf("test text is %d runes, want 300", n)
	}
	e := NewChunkedEmbedder(rec, 400)
	got, err := e.Embed(context.Background(), text)
	if err != nil {
		t.Fatalf("a text inside the budget but rejected by the model must be split, got: %v", err)
	}
	if len(got) != 8 {
		t.Fatalf("pooled vector has %d dims, want 8", len(got))
	}
	if m := rec.maxAccepted(); m > 80 {
		t.Errorf("the model accepted a %d-rune text after the retry, want at most 80", m)
	}
}

// The production shape: a text longer than the budget is cut into pieces the model
// takes, so a resource that used to be rejected is stored instead. Three pieces are
// rejected at 400 runes and each costs one halving, which is exactly the budget.
func TestChunkedEmbedderEmbedsLongTextInPieces(t *testing.T) {
	inner := &lengthLimitedEmbedder{limit: 200, dim: 6, inner: additiveEmbedder{dim: 6}}
	e := NewChunkedEmbedder(inner, 400)
	text := strings.Repeat("这是一段比较长的记忆内容。", 100) // 1300 runes → 400+400+400+100
	got, err := e.Embed(context.Background(), text)
	if err != nil {
		t.Fatalf("1300-rune text must be stored, got: %v", err)
	}
	if len(got) != 6 {
		t.Fatalf("pooled vector has %d dims, want 6", len(got))
	}
	if m := inner.maxAccepted(); m > 200 {
		t.Errorf("the model accepted a %d-rune text, want at most its 200", m)
	}
	if inner.calls < 4 {
		t.Errorf("inner was called %d times; a 1300-rune text needs several pieces", inner.calls)
	}
}

// A rejection that is not about length (the provider answers the same 400 for an empty
// or missing "input") must not become an ever-shrinking retry storm: the halving budget
// belongs to the whole call, so the work stops after a bounded number of requests and
// the caller sees the provider's own error.
func TestChunkedEmbedderGivesUpAfterTheWholeCallBudget(t *testing.T) {
	rec := &recordingEmbedder{dim: 4, fn: func(_ context.Context, s string) ([]float32, error) {
		return nil, errors.New("embed HTTP 400: The parameter is invalid. Please check again.")
	}}
	e := NewChunkedEmbedder(rec, 400)
	_, err := e.Embed(context.Background(), strings.Repeat("这是一条很长的记忆。", 100)) // 1000 runes
	if err == nil {
		t.Fatal("a model that rejects everything must produce an error")
	}
	// One request per piece tried plus the halvings the budget allows: the descent is
	// a path, not a tree, so the count stays tiny instead of shrinking to 1 rune.
	if len(rec.texts) > 10 {
		t.Errorf("the call made %d requests, want a bounded number (budget = pieces × depth)", len(rec.texts))
	}
	msg := err.Error()
	if !strings.Contains(msg, "The parameter is invalid") {
		t.Errorf("the provider's own error must survive, got: %v", err)
	}
	if !strings.Contains(msg, "1000") || !strings.Contains(msg, "recording") {
		t.Errorf("the error must carry the rune count and the model name, got: %v", err)
	}
	smallest := 1 << 30
	for _, x := range rec.texts {
		if n := utf8.RuneCountInString(x); n < smallest {
			smallest = n
		}
	}
	if smallest < embedChunkMinRunes/2 {
		t.Errorf("the smallest request was %d runes: the text was shrunk past the %d floor",
			smallest, embedChunkMinRunes/2)
	}
}

// A long, token-dense text whose every piece needs halving must still be stored: the
// budget scales with the piece count, so a piece is never starved by the pieces before
// it (the failure mode a flat whole-call budget would have).
func TestChunkedEmbedderStoresLongTokenDenseText(t *testing.T) {
	inner := &lengthLimitedEmbedder{limit: 200, dim: 6, inner: additiveEmbedder{dim: 6}}
	e := NewChunkedEmbedder(inner, 400)
	// Five pieces of 400 runes, each refused at 200 → five halvings needed, which a
	// flat budget of 3 could not pay.
	text := strings.Repeat("这是一段非常密集的长记忆内容。", 125)
	if n := utf8.RuneCountInString(text); n <= 1600 || n > 2000 {
		t.Fatalf("test text is %d runes, want 1601..2000 so it cuts into 5 pieces", n)
	}
	if got := len(splitEmbedChunks(text, 400)); got != 5 {
		t.Fatalf("test text cuts into %d pieces, want 5", got)
	}
	got, err := e.Embed(context.Background(), text)
	if err != nil {
		t.Fatalf("a 2000-rune token-dense text must be stored, got: %v", err)
	}
	if len(got) != 6 {
		t.Fatalf("pooled vector has %d dims, want 6", len(got))
	}
	if m := inner.maxAccepted(); m > 200 {
		t.Errorf("the model accepted a %d-rune text, want at most its 200", m)
	}
}

// The budget is finite but per piece: enough halvings for every piece to reach the
// floor, capped at the piece cap.
func TestSplitBudgetIsBoundedAndScales(t *testing.T) {
	// chunkChars 400 → 400/64 → 3 halvings per piece at most.
	e := NewChunkedEmbedder(NewLocalEmbedder(384), 400)
	if got := e.splitBudget(1); got != embedSplitBudget {
		t.Errorf("splitBudget(1) = %d, want the %d single-piece floor", got, embedSplitBudget)
	}
	if got := e.splitBudget(4); got != 12 {
		t.Errorf("splitBudget(4) = %d, want 4 pieces × 3 halvings", got)
	}
	if got := e.splitBudget(embedChunkMax * 10); got != embedChunkMax {
		t.Errorf("splitBudget must be capped at %d, got %d", embedChunkMax, got)
	}
	if got := e.splitBudget(0); got != embedSplitBudget {
		t.Errorf("splitBudget(0) = %d, want the single-piece floor", got)
	}
	// A chunk budget below the floor must not give a zero or negative depth.
	tiny := NewChunkedEmbedder(NewLocalEmbedder(384), 4)
	if got := tiny.splitBudget(3); got != embedSplitBudget {
		t.Errorf("a 4-rune chunk budget gave %d halvings, want the %d floor", got, embedSplitBudget)
	}
}

// The strictest measured tier: every piece needs two halvings (400 → 200 → 100 against a
// 150-rune limit). Four such pieces need eight halvings, which a flat budget of 3 — or
// "3 + pieces" = 7 — could not pay.
func TestChunkedEmbedderStoresTextWhenEveryPieceNeedsTwoHalvings(t *testing.T) {
	inner := &lengthLimitedEmbedder{limit: 150, dim: 6, inner: additiveEmbedder{dim: 6}}
	e := NewChunkedEmbedder(inner, 400)
	text := strings.Repeat("这是一段很密集的长记忆内容。", 100) // 1500 runes → 4 pieces
	if n := utf8.RuneCountInString(text); n <= 1200 || n > 1600 {
		t.Fatalf("test text is %d runes, want 1201..1600 so it cuts into 4 pieces", n)
	}
	if got := len(splitEmbedChunks(text, 400)); got != 4 {
		t.Fatalf("test text cuts into %d pieces, want 4", got)
	}
	if _, err := e.Embed(context.Background(), text); err != nil {
		t.Fatalf("four pieces needing two halvings each must still be stored, got: %v", err)
	}
	if m := inner.maxAccepted(); m > 150 {
		t.Errorf("the model accepted a %d-rune text, want at most its 150", m)
	}
}

// The extractor accepts 30,000 characters, so the splitter must be able to carry that
// much: otherwise a memory the write path takes can never be stored. Measured before the
// cap was raised: 25,600 runes passed (63 pieces), 30,000 failed (76 pieces) with
// "needs 76 chunks of at most 400 runes …, over the 64-chunk cap".
func TestChunkedEmbedderStoresTheExtractionLimit(t *testing.T) {
	inner := &lengthLimitedEmbedder{limit: 510, dim: 6, inner: additiveEmbedder{dim: 6}}
	e := NewChunkedEmbedder(inner, defaultEmbedChunkChars)
	unit := "这是用于验证嵌入分片行为的中文长文本。" // 19 runes
	text := strings.Repeat(unit, extractMaxChars/utf8.RuneCountInString(unit)+1)
	n := utf8.RuneCountInString(text)
	if n < extractMaxChars {
		t.Fatalf("test text is %d runes, want at least the %d extraction limit", n, extractMaxChars)
	}
	pieces := len(splitEmbedChunks(text, defaultEmbedChunkChars))
	if pieces > embedChunkMax {
		t.Fatalf("%d runes cut into %d pieces, over the %d cap the extractor's limit needs", n, pieces, embedChunkMax)
	}
	if _, err := e.Embed(context.Background(), text); err != nil {
		t.Fatalf("a %d-rune memory (the extraction limit) must be stored, got: %v", n, err)
	}
	if m := inner.maxAccepted(); m > defaultEmbedChunkChars {
		t.Errorf("the model accepted a %d-rune text, want at most the %d budget", m, defaultEmbedChunkChars)
	}
}

// The startup check: a chunk budget that cannot cover the extraction limit must say so,
// because the failure otherwise appears only as a 500 on one long write.
func TestEmbedChunkCapWarning(t *testing.T) {
	if w := embedChunkCapWarning(defaultEmbedChunkChars); w != "" {
		t.Errorf("the default budget must cover extractMaxChars, got %q", w)
	}
	w := embedChunkCapWarning(200)
	if w == "" {
		t.Fatal("a 200-rune budget needs 150 pieces for extractMaxChars and must warn")
	}
	if !strings.Contains(w, embedChunkCharsEnv) || !strings.Contains(w, "375") {
		t.Errorf("the warning must name the knob and the minimum that works (375), got %q", w)
	}
	if w := embedChunkCapWarning(0); w != "" {
		t.Errorf("a zero budget is the caller's default, not a warning: %q", w)
	}
}

// maxAccepted is the longest text the model actually took, which is what the
// decorator's splitting is judged on.
func (l *lengthLimitedEmbedder) maxAccepted() int {
	m := 0
	for _, t := range l.accepted {
		if n := utf8.RuneCountInString(t); n > m {
			m = n
		}
	}
	return m
}

// A text that cannot be stored within the piece cap fails with an error that says how
// long it was and which model refused it — never a bare status code.
func TestChunkedEmbedderTooManyChunksNamesTheSize(t *testing.T) {
	inner := &lengthLimitedEmbedder{limit: 5, dim: 4, inner: additiveEmbedder{dim: 4}}
	e := NewChunkedEmbedder(inner, 4)
	text := strings.Repeat("x", 500)
	_, err := e.Embed(context.Background(), text)
	if err == nil {
		t.Fatal("a 500-rune text at a 4-rune budget must hit the chunk cap")
	}
	if !strings.Contains(err.Error(), "500") || !strings.Contains(err.Error(), "bge-large-zh-v1.5") {
		t.Errorf("error must name the rune count and the model, got: %v", err)
	}
}

// When every split still fails, the error carries the rune length and the model name
// (issue #24: the old error was a bare HTTP code on a path that answered 500).
func TestChunkedEmbedderErrorCarriesLengthAndModel(t *testing.T) {
	rec := &recordingEmbedder{dim: 4, fn: func(_ context.Context, s string) ([]float32, error) {
		return nil, errors.New("embed HTTP 400: refusal")
	}}
	e := NewChunkedEmbedder(rec, 400)
	text := strings.Repeat("字", 300)
	_, err := e.Embed(context.Background(), text)
	if err == nil {
		t.Fatal("an always-rejecting model must produce an error")
	}
	if !strings.Contains(err.Error(), "300") || !strings.Contains(err.Error(), "recording") {
		t.Errorf("error must carry the rune count and the model name, got: %v", err)
	}
}

// Only a "too long" rejection is worth splitting: a dead endpoint must fail at once
// instead of multiplying requests.
func TestChunkedEmbedderDoesNotSplitOnOtherErrors(t *testing.T) {
	rec := &recordingEmbedder{dim: 4, fn: func(_ context.Context, s string) ([]float32, error) {
		return nil, errors.New("dial tcp 127.0.0.1:1: connect: connection refused")
	}}
	e := NewChunkedEmbedder(rec, 400)
	if _, err := e.Embed(context.Background(), strings.Repeat("字", 300)); err == nil {
		t.Fatal("a dead endpoint must produce an error")
	}
	if len(rec.texts) != 1 {
		t.Errorf("inner was called %d times, want 1 (no splitting for a non-length error)", len(rec.texts))
	}
}

// A dimension mismatch is not a length problem: splitting cannot fix it.
func TestLooksLikeInputTooLongExcludesDimMismatch(t *testing.T) {
	if looksLikeInputTooLong(errors.New("couldn't calculate similarity: vectors must have the same length")) {
		t.Error("a chromem dimension mismatch must not be answered by splitting the text")
	}
	if !looksLikeInputTooLong(errors.New("embed HTTP 400 (model m, 10 runes): {\"code\":20015}")) {
		t.Error("a 400 must be treated as a possible length rejection")
	}
	// The real remote error truncates the body to 200 bytes, so the nested code is
	// usually cut off: only the status prefix can be relied on.
	if !looksLikeInputTooLong(errors.New(`embed HTTP 400 (model m, 1500 runes): {"error":{"code":null,"message":"400 Bad Request: {"code":20015","message":"The parameter is invalid. Please check again."`)) {
		t.Error("a truncated 400 body must still be treated as a possible length rejection")
	}
	if !looksLikeInputTooLong(errors.New("embed HTTP 413: payload too large")) {
		t.Error("a 413 must be treated as a possible length rejection")
	}
	if looksLikeInputTooLong(nil) {
		t.Error("nil is not a length error")
	}
}

// The decorator must not hide the wrapped embedder's capabilities: chunking does not
// change the dimension, and a hashed local stub stays non-semantic.
func TestChunkedEmbedderForwardsDimAndSemantic(t *testing.T) {
	local := NewChunkedEmbedder(NewLocalEmbedder(384), 400)
	if got := local.Dim(); got != 384 {
		t.Errorf("Dim() = %d, want the inner 384", got)
	}
	if isSemantic(local) {
		t.Error("the wrapped local stub must still report non-semantic vectors")
	}
	if _, ok := local.Unwrap().(*LocalEmbedder); !ok {
		t.Error("Unwrap must expose the wrapped embedder")
	}
	remote := NewChunkedEmbedder(NewOpenAIEmbedder("http://127.0.0.1:1/v1", "", "m"), 400)
	if !isSemantic(remote) {
		t.Error("a wrapped remote embedder is semantic")
	}
	if got := embedderModelName(remote); got != "m" {
		t.Errorf("model name = %q, want m", got)
	}
}

// needsEmbedderProbe was a type assertion on *OpenAIEmbedder, which a decorator makes
// permanently false: the remote readiness wait (fix 3) would disappear while the
// existing test, which builds a bare embedder, stayed green.
func TestNeedsEmbedderProbeSeesThroughTheDecorator(t *testing.T) {
	wrapped := &MemoryStore{embed: NewChunkedEmbedder(NewOpenAIEmbedder("http://127.0.0.1:1/v1", "", "m"), 400)}
	if !wrapped.needsEmbedderProbe() {
		t.Error("a decorated remote embedder must still trigger the readiness probe")
	}
	plain := &MemoryStore{embed: NewChunkedEmbedder(NewLocalEmbedder(384), 400)}
	if plain.needsEmbedderProbe() {
		t.Error("a decorated local stub must not trigger the readiness probe")
	}
}

// The write path that produced the 500: a long memory must be stored, and its vector
// must have the embedder's dimension. The model's boundary is the live one (510 runes
// passed, 511 did not), so no piece may be sent over 400.
func TestStoreStoresLongTextWithChunkedEmbedder(t *testing.T) {
	dir := t.TempDir()
	inner := &lengthLimitedEmbedder{limit: 510, dim: 384, inner: NewLocalEmbedder(384)}
	st, err := NewMemoryStore(ctxForTest(), dir, NewChunkedEmbedder(inner, 400),
		filepath.Join(dir, "graph.json"), 384)
	if err != nil {
		t.Skipf("MemoryStore unavailable in test env: %v", err)
	}
	defer st.Close()

	id := newID()
	long := strings.Repeat("这是一条很长的记忆。", 200) // 2000 runes, over the old 400
	if err := st.Add(memory.L3Fact, id, long, map[string]string{"user_id": "u", "agent_id": "a"}); err != nil {
		t.Fatalf("a 2000-rune memory must be stored, got: %v", err)
	}
	got := st.GetMany([]string{id})
	if len(got) != 1 || got[0].Content != long {
		t.Fatalf("stored row: %+v", got)
	}
	if m := inner.maxAccepted(); m > 400 {
		t.Errorf("the model accepted a %d-rune text, want at most the 400 budget", m)
	}
}

// The budget knob: default, override, and junk (which must not silence the splitter).
func TestResolveEmbedChunkChars(t *testing.T) {
	t.Setenv(embedChunkCharsEnv, "")
	if got := resolveEmbedChunkChars(); got != defaultEmbedChunkChars {
		t.Errorf("unset = %d, want the %d default", got, defaultEmbedChunkChars)
	}
	t.Setenv(embedChunkCharsEnv, "256")
	if got := resolveEmbedChunkChars(); got != 256 {
		t.Errorf("override = %d, want 256", got)
	}
	t.Setenv(embedChunkCharsEnv, "0")
	if got := resolveEmbedChunkChars(); got != defaultEmbedChunkChars {
		t.Errorf("zero = %d, want the default (chunking must not be switchable off)", got)
	}
	t.Setenv(embedChunkCharsEnv, "big")
	if got := resolveEmbedChunkChars(); got != defaultEmbedChunkChars {
		t.Errorf("junk = %d, want the default", got)
	}
}

// A non-positive chunk budge means the default, never "send everything whole".
func TestNewChunkedEmbedderRejectsNonPositiveBudget(t *testing.T) {
	if got := NewChunkedEmbedder(NewLocalEmbedder(384), 0).chunkChars; got != defaultEmbedChunkChars {
		t.Errorf("chunkChars = %d, want the default", got)
	}
}
