package main

import (
	"context"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/philippgille/chromem-go"
	"github.com/tuancookiez-hub/hyatlas-v4/memory"
)

// Re-embedding every stored document is what a model change needs: vectors from two
// different models are not comparable, so a collection written before the change is
// half in one vector space and half in another, and similarity between them is noise.
// Until now the only way to fix that was the offline tool in scripts/reembed, which
// needs the server stopped and cannot see the clause rows at all (they are not in
// doc_index.json, which is what it walks).
//
// This pass does the same work with the server running, under the same document ID:
// chromem's AddDocument overwrites the row (map entry + gob file) in place, so nothing
// is deleted, the pass is interrupted or repeated safely, and metadata and content are
// carried over unchanged.

// ReembedReport is what one pass did (or, for a dry run, would do).
type ReembedReport struct {
	Done   int            `json:"done"`
	Total  int            `json:"total"`
	Failed int            `json:"failed"`
	Errors []string       `json:"errors"`
	Layers map[string]int `json:"layers"`
	DryRun bool           `json:"dry_run"`
}

// reembedErrorSample caps how many failures travel back in the response: a pass over a
// broken endpoint would otherwise return one message per document.
const reembedErrorSample = 20

// reembedTargets validates the collection names a request asked for, defaulting to
// every collection the store owns. A name that is not one of them is an error rather
// than an empty pass, so a typo cannot read as "nothing to do".
func reembedTargets(want []string) ([]string, error) {
	all := allCollections()
	if len(want) == 0 {
		return all, nil
	}
	known := make(map[string]bool, len(all))
	for _, n := range all {
		known[n] = true
	}
	out := make([]string, 0, len(want))
	seen := make(map[string]bool, len(want))
	for _, w := range want {
		n := strings.TrimSpace(w)
		if n == "" {
			continue
		}
		if !known[n] {
			return nil, fmt.Errorf("unknown collection %q; known collections: %s", n, strings.Join(all, ", "))
		}
		if !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no collection named; known collections: %s", strings.Join(all, ", "))
	}
	return out, nil
}

// listCollectionDocs returns every document of one collection.
//
// chromem has no "list all documents" call, so this enumerates the way rebuildIndex
// does: a query with a zero vector scores every document equally and returns them all.
// The query vector's length must equal the stored vectors' length, so the store's own
// dimension is tried first and the embedder's answer second — that tolerance is what
// lets a pass run at all when the configured dimension and the stored one disagree,
// which is the situation a model change creates.
func (s *MemoryStore) listCollectionDocs(ctx context.Context, name string, col *chromem.Collection) ([]chromem.Result, error) {
	n := col.Count()
	if n == 0 {
		return nil, nil
	}
	var tried []int
	var lastErr error
	for _, dim := range []int{s.dimFor(memory.Layer(name)), s.embed.Dim(), embedDim, 1024} {
		if dim <= 0 || slicesContains(tried, dim) {
			continue
		}
		tried = append(tried, dim)
		res, err := col.QueryEmbedding(ctx, make([]float32, dim), n, nil, nil)
		if err == nil {
			return res, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("couldn't enumerate %s (%d docs): %w (tried dimension(s) %v)", name, n, lastErr, tried)
}

// slicesContains is a tiny membership check; the candidate list is three or four
// entries, so its size is never the point.
func slicesContains(xs []int, x int) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

// ReembedAll re-embeds every document of the named collections with the store's
// current embedder, writing each row back under its own ID.
//
// It is deliberately not atomic and never deletes: a pass that stops halfway leaves a
// collection half in the old space, and calling it again finishes the job. A dry run
// still embeds every document (that is how it finds out whether the pass would work)
// and writes nothing.
//
// A document whose new vector length differs from its stored one is reported and left
// alone: chromem compares every pair of vectors by length, so writing one 1024d vector
// into a 1024d collection that this model no longer matches would break every query on
// it. Changing a model's dimension means rebuilding the collection, not mixing
// lengths inside it.
func (s *MemoryStore) ReembedAll(ctx context.Context, names []string, limit int, dryRun bool) ReembedReport {
	rep := ReembedReport{Errors: []string{}, Layers: map[string]int{}, DryRun: dryRun}
	fail := func(format string, args ...any) {
		rep.Failed++
		if len(rep.Errors) < reembedErrorSample {
			rep.Errors = append(rep.Errors, fmt.Sprintf(format, args...))
		}
	}
	for _, name := range names {
		if limit > 0 && rep.Total >= limit {
			break
		}
		col := s.byName[name]
		if col == nil {
			fail("no collection named %s", name)
			continue
		}
		docs, err := s.listCollectionDocs(ctx, name, col)
		if err != nil {
			fail("%v", err)
			continue
		}
		for _, d := range docs {
			if limit > 0 && rep.Total >= limit {
				break
			}
			if ctx.Err() != nil {
				fail("pass cancelled: %v", ctx.Err())
				return rep
			}
			rep.Total++
			rep.Layers[name]++
			text := d.Content
			if strings.TrimSpace(text) == "" {
				fail("%s/%s: no content to re-embed (%d-rune vector stored)", name, d.ID, len(d.Embedding))
				continue
			}
			emb, err := s.embed.Embed(ctx, text)
			if err != nil {
				fail("%s/%s (%d runes): %v", name, d.ID, utf8.RuneCountInString(text), err)
				continue
			}
			if len(emb) != len(d.Embedding) {
				fail("%s/%s: the embedder now returns %d dimensions but this collection holds %d; "+
					"refusing to mix vector lengths (rebuild this collection instead)",
					name, d.ID, len(emb), len(d.Embedding))
				continue
			}
			if !dryRun {
				// rowMu keeps this rewrite apart from a Supersede rewrite of the same
				// row, exactly as the write path does.
				s.rowMu.Lock()
				err = col.AddDocument(ctx, chromem.Document{
					ID: d.ID, Content: d.Content, Embedding: emb, Metadata: d.Metadata,
				})
				s.rowMu.Unlock()
				if err != nil {
					fail("%s/%s: %v", name, d.ID, err)
					continue
				}
			}
			rep.Done++
		}
	}
	return rep
}
