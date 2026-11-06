package main

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

// promptCase is one system prompt the server can send, named for the failure message.
type promptCase struct {
	name string
	text string
}

// systemPrompts lists every system prompt, including both branches of the extraction
// prompt. It is a function rather than a local slice so the tripwire below can ask how
// many prompts the test knows about.
func systemPrompts() []promptCase {
	return []promptCase{
		{"extract (one turn)", extractSystemPrompt(false)},
		{"extract (with context)", extractSystemPrompt(true)},
		{"reconcile", reconcileSystemPrompt},
		{"consolidate", consolidateSystem("", "", "")},
		{"cross-schema", crossSchemaSystem(2, "- m1 | the deploy window moved to friday")},
		{"episode", episodeSystemPrompt},
		{"now", nowSystemPrompt},
		{"backlog", backlogSystem},
	}
}

// TestEverySystemPromptStatesTheLanguage checks that every system prompt the server
// sends carries the language rule. Without it a small model mirrors the prompt's own
// English and writes the store in English; a prompt added without the rule would
// silently do that again.
func TestEverySystemPromptStatesTheLanguage(t *testing.T) {
	for _, p := range systemPrompts() {
		if !strings.Contains(p.text, "Language rules:") {
			t.Errorf("%s prompt lacks the language rule", p.name)
		}
		if !strings.Contains(p.text, "简体中文") {
			t.Errorf("%s prompt does not name Simplified Chinese", p.name)
		}
	}
}

// systemCallSites is how many places in this package's non-test sources send a system
// message. It is a number a human must bump, on purpose: the list above cannot notice a
// prompt nobody listed, which is how the cross-schema prompt kept writing English while
// this test stayed green. The extraction call site produces two prompts from one site
// (with and without context), so the list is one longer than this count.
const systemCallSites = 7

// TestNoSystemPromptEscapesTheList is the tripwire for the next prompt: it counts the
// system-role literals in the package's own sources and fails when the sources send more
// system messages than the list above accounts for.
func TestNoSystemPromptEscapesTheList(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	got := 0
	var seen []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if n := strings.Count(string(data), `{"role": "system"`); n > 0 {
			got += n
			seen = append(seen, fmt.Sprintf("%s(%d)", name, n))
		}
	}
	if got != systemCallSites {
		t.Fatalf("found %d system-role call site(s) %v, want %d — 你新增了一个系统提示词：把它加进 systemPrompts() 清单，并确认它拼了 languageRule", got, seen, systemCallSites)
	}
	if len(systemPrompts()) != systemCallSites+1 {
		t.Fatalf("systemPrompts() has %d entries, want %d (one call site sends two prompts: extract with and without context)", len(systemPrompts()), systemCallSites+1)
	}
}

// TestCrossSchemaSystemKeepsItsContract checks the rest of the cross-schema prompt
// survived the move into a function: the JSON shape the parser expects and the material
// the model is asked to generalise.
func TestCrossSchemaSystemKeepsItsContract(t *testing.T) {
	const material = "- m1 | the deploy window moved to friday"
	got := crossSchemaSystem(3, material)
	for _, want := range []string{"Language rules:", "简体中文", `{"patterns":`, "at most 3 NEW patterns", "Schemas:", material} {
		if !strings.Contains(got, want) {
			t.Errorf("cross-schema prompt no longer contains %q", want)
		}
	}
	if strings.Index(got, "Language rules:") > strings.Index(got, "Schemas:") {
		t.Error("the language rule must precede the schema list, as it does in the consolidate prompt")
	}
}

// TestLanguageRuleKeepsCodeVerbatim checks the other half of the rule: the same
// instruction that asks for Chinese prose has to protect identifiers, or the model
// starts translating env var names and ports along with the prose.
func TestLanguageRuleKeepsCodeVerbatim(t *testing.T) {
	for _, want := range []string{"identifiers", "file paths", "environment variable names", "Never translate"} {
		if !strings.Contains(languageRule, want) {
			t.Errorf("languageRule no longer protects %q", want)
		}
	}
}
