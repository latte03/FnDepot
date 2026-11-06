package main

import (
	"strings"
	"testing"
)

// TestEverySystemPromptStatesTheLanguage checks that every system prompt the server
// sends carries the language rule. Without it a small model mirrors the prompt's own
// English and writes the store in English; a prompt added without the rule would
// silently do that again, so every prompt is listed here, including both branches of
// the extraction prompt.
func TestEverySystemPromptStatesTheLanguage(t *testing.T) {
	prompts := []struct {
		name string
		text string
	}{
		{"extract (one turn)", extractSystemPrompt(false)},
		{"extract (with context)", extractSystemPrompt(true)},
		{"reconcile", reconcileSystemPrompt},
		{"consolidate", consolidateSystem("", "", "")},
		{"episode", episodeSystemPrompt},
		{"now", nowSystemPrompt},
		{"backlog", backlogSystem},
	}
	for _, p := range prompts {
		if !strings.Contains(p.text, "Language rules:") {
			t.Errorf("%s prompt lacks the language rule", p.name)
		}
		if !strings.Contains(p.text, "简体中文") {
			t.Errorf("%s prompt does not name Simplified Chinese", p.name)
		}
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
