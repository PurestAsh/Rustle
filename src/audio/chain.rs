//! Audio processing chain
//!
//! Unified audio processing pipeline that combines:
//! - Preamp (gain control before EQ)
//! - 10-band parametric equalizer
//! - Fade envelope
//! - Real-time audio analyzer for visualization
//!

use rodio::Source;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use super::analyzer::{AnalyzingSource, AudioAnalysisData};
use super::equalizer::{Equalizer, EqualizerParams};
use super::fade::{FadeControl, FadeEnvelope};

/// Shared audio processing chain parameters
///
/// This struct holds all audio processing parameters and can be cloned
/// to share between AudioPlayer and UI. All parameters are thread-safe
/// and can be updated in real-time.
#[derive(Clone)]
pub struct AudioProcessingChain {
    preamp_linear_bits: Arc<AtomicU32>,
    /// Equalizer parameters (has its own Arc<RwLock>)
    eq_params: EqualizerParams,
    /// Fade control for smooth volume transitions
    fade_control: FadeControl,
    /// Audio analysis data for visualization
    analysis: AudioAnalysisData,
}

impl AudioProcessingChain {
    /// Create a new audio processing chain
    pub fn new() -> Self {
        Self {
            preamp_linear_bits: Arc::new(AtomicU32::new(1.0_f32.to_bits())),
            eq_params: EqualizerParams::new(44100),
            fade_control: FadeControl::new(1.0),
            analysis: AudioAnalysisData::new(),
        }
    }
    pub fn fade_to(&self, volume: f32, duration: std::time::Duration) {
        self.fade_control.fade_to(volume, duration);
    }

    /// Set volume instantly
    pub fn set_fade_volume(&self, volume: f32) {
        self.fade_control.set_volume(volume);
    }

    // ========================================================================
    // Preamp controls
    // ========================================================================

    /// Set preamp gain in dB (-12 to +12)
    pub fn set_preamp(&self, db: f32) {
        let db = db.clamp(-12.0, 12.0);
        let linear = if db.abs() < 0.01 {
            1.0
        } else {
            10.0_f32.powf(db / 20.0)
        };
        self.preamp_linear_bits
            .store(linear.to_bits(), Ordering::Release);
    }

    // ========================================================================
    // Equalizer controls
    // ========================================================================

    /// Enable or disable the equalizer
    pub fn set_equalizer_enabled(&self, enabled: bool) {
        self.eq_params.set_enabled(enabled);
    }

    /// Set all 10 band gains at once (in dB, typically -12 to +12)
    pub fn set_equalizer_gains(&self, gains: [f32; 10]) {
        self.eq_params.set_gains(gains);
    }

    // ========================================================================
    // Analysis data access
    // ========================================================================

    /// Get audio analysis data for visualization
    pub fn analysis(&self) -> &AudioAnalysisData {
        &self.analysis
    }

    /// Reset analysis data (call when playback stops)
    pub fn reset_analysis(&self) {
        self.analysis.reset();
    }

    pub fn set_analysis_enabled(&self, enabled: bool) {
        self.analysis.set_enabled(enabled);
    }

    /// Force EQ coefficients refresh
    /// This marks the EQ parameters as dirty, forcing a recalculation
    /// on the next audio sample. Useful when switching tracks to ensure
    /// the audio processing chain is properly initialized.
    pub fn refresh_eq_coefficients(&self) {
        self.eq_params.mark_dirty();
    }

    // ========================================================================
    // Chain configuration
    // ========================================================================

    /// Update sample rate (called when audio format changes)
    pub fn set_sample_rate(&self, sample_rate: u32) {
        self.eq_params.set_sample_rate(sample_rate);
    }

    // ========================================================================
    // Source processing
    // ========================================================================

    /// Apply the processing chain to an audio source
    ///
    /// Processing order:
    /// 1. Preamp (gain adjustment)
    /// 2. Equalizer (10-band parametric EQ)
    /// 3. Analyzer (for visualization, doesn't modify audio)
    pub fn apply<S>(&self, source: S) -> ProcessedSource<S>
    where
        S: Source<Item = f32>,
    {
        // Update sample rate from source
        self.set_sample_rate(source.sample_rate());

        ProcessedSource::new(source, self.clone())
    }
}

impl Default for AudioProcessingChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio source with processing chain applied
///
/// This wraps the source and applies preamp, EQ, fade, and analysis in sequence.
pub struct ProcessedSource<S>
where
    S: Source<Item = f32>,
{
    /// Inner source with full processing chain applied
    inner: AnalyzingSource<FadeEnvelope<Equalizer<PreampSource<S>>>>,
}

impl<S> ProcessedSource<S>
where
    S: Source<Item = f32>,
{
    fn new(source: S, chain: AudioProcessingChain) -> Self {
        // Build processing chain: Source -> Preamp -> EQ -> Fade -> Analyzer
        let preamp_source = PreampSource::new(source, chain.preamp_linear_bits.clone());
        let eq_source = Equalizer::new(preamp_source, chain.eq_params.clone());
        let fade_source = FadeEnvelope::new(eq_source, chain.fade_control.clone());
        let analyzed = AnalyzingSource::new(fade_source, chain.analysis.clone());

        Self { inner: analyzed }
    }
}

impl<S> Iterator for ProcessedSource<S>
where
    S: Source<Item = f32>,
{
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<S> Source for ProcessedSource<S>
where
    S: Source<Item = f32>,
{
    fn current_span_len(&self) -> Option<usize> {
        self.inner.current_span_len()
    }

    fn channels(&self) -> u16 {
        self.inner.channels()
    }

    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    fn total_duration(&self) -> Option<std::time::Duration> {
        self.inner.total_duration()
    }

    fn try_seek(&mut self, pos: std::time::Duration) -> Result<(), rodio::source::SeekError> {
        self.inner.try_seek(pos)
    }
}

/// Preamp source wrapper that applies gain before other processing
struct PreampSource<S>
where
    S: Source<Item = f32>,
{
    source: S,
    preamp_linear_bits: Arc<AtomicU32>,
}

impl<S> PreampSource<S>
where
    S: Source<Item = f32>,
{
    fn new(source: S, preamp_linear_bits: Arc<AtomicU32>) -> Self {
        Self {
            source,
            preamp_linear_bits,
        }
    }

    fn preamp_linear(&self) -> f32 {
        f32::from_bits(self.preamp_linear_bits.load(Ordering::Acquire))
    }
}

impl<S> Iterator for PreampSource<S>
where
    S: Source<Item = f32>,
{
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let sample = self.source.next()?;
        let gain = self.preamp_linear();

        if (gain - 1.0).abs() < 0.001 {
            Some(sample)
        } else {
            // Apply gain with soft clipping to prevent harsh distortion
            let amplified = sample * gain;
            Some(soft_clip(amplified))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.source.size_hint()
    }
}

impl<S> Source for PreampSource<S>
where
    S: Source<Item = f32>,
{
    fn current_span_len(&self) -> Option<usize> {
        self.source.current_span_len()
    }

    fn channels(&self) -> u16 {
        self.source.channels()
    }

    fn sample_rate(&self) -> u32 {
        self.source.sample_rate()
    }

    fn total_duration(&self) -> Option<std::time::Duration> {
        self.source.total_duration()
    }

    fn try_seek(&mut self, pos: std::time::Duration) -> Result<(), rodio::source::SeekError> {
        self.source.try_seek(pos)
    }
}

/// Soft clipping function to prevent harsh digital clipping
fn soft_clip(x: f32) -> f32 {
    if x.abs() < 0.9 {
        x
    } else if x > 0.0 {
        0.9 + 0.1 * ((x - 0.9) / 0.1).tanh()
    } else {
        -0.9 - 0.1 * ((-x - 0.9) / 0.1).tanh()
    }
}
