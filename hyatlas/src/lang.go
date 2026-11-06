package main

// languageRule tells the model which language to write prose in. The server's
// prompts are English, so without it a small model mirrors the prompt language
// and writes the store in English (see the 2026-11 language audit).
//
// It is appended to every system prompt the server sends, so the store reads in
// the user's language no matter which pass wrote a row. lang_test.go checks that
// each prompt carries it; a new prompt without it would quietly write English
// again.
const languageRule = `Language rules:
- Write every human-readable field in Simplified Chinese (简体中文), even when the source text is English.
- Keep identifiers, file paths, commands, environment variable names, error messages, model names, product names and code exactly as they appear. Never translate, transliterate or reformat them.
- JSON keys, layer names and enum values stay exactly as specified, in English.`
