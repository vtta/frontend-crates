// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::ops::{Deref, Range, RangeTo};

/// Buffered parser text with the byte ranges of complete control tokens.
/// `None` retains the legacy text-only matching contract.
#[derive(Debug, Default)]
pub(crate) struct InputBuffer {
    text: String,
    controls: Option<Vec<Range<usize>>>,
}

impl InputBuffer {
    pub(crate) fn tokens() -> Self {
        Self {
            text: String::new(),
            controls: Some(Vec::new()),
        }
    }

    pub(crate) fn is_token_input(&self) -> bool {
        self.controls.is_some()
    }

    pub(crate) fn view(&self) -> InputText<'_> {
        InputText {
            text: &self.text,
            controls: self.controls.as_deref(),
            offset: 0,
        }
    }

    pub(crate) fn push_str(&mut self, text: &str) {
        self.text.push_str(text);
    }

    pub(crate) fn push_control(&mut self, text: &str) -> anyhow::Result<()> {
        let controls = self
            .controls
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("control tokens require token input"))?;
        if !text.is_empty() {
            controls.push(self.text.len()..self.text.len() + text.len());
        }
        self.text.push_str(text);
        Ok(())
    }

    pub(crate) fn drain(&mut self, range: RangeTo<usize>) -> std::string::Drain<'_> {
        if let Some(controls) = &mut self.controls {
            controls.retain_mut(|r| {
                if r.start < range.end {
                    return false;
                }
                r.start -= range.end;
                r.end -= range.end;
                true
            });
        }
        self.text.drain(range)
    }

    pub(crate) fn prefix_owned(&self, end: usize) -> Self {
        Self {
            text: self.text[..end].to_owned(),
            controls: self
                .controls
                .as_ref()
                .map(|ranges| ranges.iter().filter(|r| r.end <= end).cloned().collect()),
        }
    }

    pub(crate) fn take_text(&mut self) -> String {
        if let Some(controls) = &mut self.controls {
            controls.clear();
        }
        std::mem::take(&mut self.text)
    }
}

impl Deref for InputBuffer {
    type Target = str;
    fn deref(&self) -> &str {
        &self.text
    }
}

/// A borrowed part of the scanner input, retaining its control-token positions.
#[derive(Clone, Copy)]
pub(crate) struct InputText<'a> {
    text: &'a str,
    controls: Option<&'a [Range<usize>]>,
    offset: usize,
}

impl<'a> InputText<'a> {
    pub(crate) fn plain(text: &'a str) -> Self {
        Self {
            text,
            controls: None,
            offset: 0,
        }
    }
    pub(crate) fn slice(self, range: Range<usize>) -> Self {
        Self {
            text: &self.text[range.clone()],
            offset: self.offset + range.start,
            ..self
        }
    }
    pub(crate) fn suffix(self, start: usize) -> Self {
        self.slice(start..self.text.len())
    }
    pub(crate) fn is_marker(self, at: usize, marker: &str) -> bool {
        self.text[at..].starts_with(marker)
            && self.controls.is_none_or(|controls| {
                controls.iter().any(|r| {
                    r.start >= self.offset + at && r.end <= self.offset + at + marker.len()
                })
            })
    }
    pub(crate) fn find_marker(self, marker: &str) -> Option<usize> {
        self.text
            .match_indices(marker)
            .map(|(at, _)| at)
            .find(|&at| self.is_marker(at, marker))
    }
    pub(crate) fn find_first(self, markers: &[String]) -> Option<(usize, usize)> {
        markers
            .iter()
            .filter_map(|marker| self.find_marker(marker).map(|at| (at, marker.len())))
            .min_by_key(|&(at, _)| at)
    }
}

impl Deref for InputText<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        self.text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn literal_markers_stay_literal_after_slicing_and_consuming() {
        let mut input = InputBuffer::tokens();
        input.push_str("中文<｜DSML｜ calls> then <");
        input.push_control("｜DSML｜").unwrap();
        input.push_str(" calls>");
        let at = "中文<｜DSML｜ calls> then ".len();
        assert_eq!(input.view().find_marker("<｜DSML｜ calls>"), Some(at));
        assert_eq!(
            input.view().suffix(at).find_marker("<｜DSML｜ calls>"),
            Some(0)
        );
        input.drain(..at).for_each(drop);
        assert_eq!(input.view().find_marker("<｜DSML｜ calls>"), Some(0));
        input.take_text();
        input.push_str("</think>");
        assert!(input.view().find_marker("</think>").is_none());
    }
    #[test]
    fn text_only_input_retains_spelling_matches() {
        let mut input = InputBuffer::default();
        input.push_str("</think>");
        assert_eq!(input.view().find_marker("</think>"), Some(0));
        assert!(input.push_control("</think>").is_err());
    }
}

impl<'a> From<&'a str> for InputText<'a> {
    fn from(text: &'a str) -> Self {
        Self::plain(text)
    }
}
