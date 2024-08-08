pub mod graph;

use fundsp::hacker::*;
use std::{cmp::Ordering, f32::consts::PI};

// note that this should be a constant because FunDSP's overlap factor is 4 (as of right now)
// this needs to match the overlap factor
const OVERSAMPLE_FACTOR: usize = 4;

// TODO:
// rewrite ALL OF this dogshit code
// refactor ALL OF THIS LOL

pub fn process(fft: &mut FftWindow, intervals: &[Shared]) {
    // extract value from shared interval variables
    let intervals: Vec<_> = intervals.iter().map(|x| x.value()).collect();
    // generate a list of in-key-frequencies every tick
    let in_key_frequencies = &generate_frequencies(&intervals);

    let window_length = fft.length();
    let step_size = window_length / OVERSAMPLE_FACTOR;
    let expected_phase = 2.0 * PI * step_size as f32 / window_length as f32;
    let freq_per_bin = fft.sample_rate() as f32 / window_length as f32;

    for channel in 0..=1 {
        let mut last_phase = vec![0.0; window_length];
        let mut phase_sum = vec![0.0; window_length];

        let mut analysis = vec![(0.0, 0.0); fft.bins()];
        let mut synthesis = vec![(0.0, 0.0); fft.bins()];

        // TODO:
        // parameterize!
        let processing_band_min = 100.0;
        let processing_band_max = 10_000.0;

        // https://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/

        // most of this code is translated from this example (c++ to rust)
        // TODO:
        // review to ensure no translation mistakes

        // https://blogs.zynaptiq.com/bernsee/repo/smbPitchShift.cpp

        /* this is the analysis step */
        for k in 0..fft.bins() {
            if !(processing_band_min..=processing_band_max).contains(&fft.frequency(k)) {
                continue;
            }

            /* compute magnitude and phase */
            let current = fft.at(channel, k);
            let phase = current.atan().re;

            /* compute phase difference */
            let mut tmp = phase - last_phase[k];
            last_phase[k] = phase;

            /* subtract expected phase difference */
            // (from overlap)
            tmp -= k as f32 * expected_phase;

            /* map delta phase into +/- Pi interval */
            let mut qpd = (tmp / PI) as i32;
            let v = qpd & 1;
            if qpd >= 0 {
                qpd += v;
            } else {
                qpd -= v;
            }
            tmp -= PI * qpd as f32;

            /* get deviation from bin frequency from the +/- Pi interval */
            tmp = OVERSAMPLE_FACTOR as f32 * tmp / (2.0 * PI);
            /* compute the k-th partials' true frequency */
            let true_freq = k as f32 * freq_per_bin + tmp * freq_per_bin;
            let magnitude = current.norm();
            /* store magnitude and true frequency in analysis array */
            analysis[k] = (true_freq, magnitude);
        }
        /* this does the actual pitch shifting */
        let mut vov: Vec<(Vec<f32>, f32)> = vec![(Vec::new(), 0.0); fft.bins()];
        for k in 0..fft.bins() {
            if !(processing_band_min..=processing_band_max).contains(&fft.frequency(k)) {
                continue;
            }

            let freq = analysis[k].0;
            let mag = analysis[k].1;

            synthesis[k].0 = freq;

            if is_in_key(in_key_frequencies, freq) {
                synthesis[k].1 += mag;
            } else {
                // TODO:
                // rework this entire stupid algorithm
                let mut nearest = (0.0, 0.0);
                let mut min_diff = f32::MAX;

                for ikf in in_key_frequencies {
                    let new_diff = (freq - ikf.0).abs();
                    if new_diff < min_diff {
                        min_diff = new_diff;
                        nearest = *ikf;
                    }
                }

                let mut nearest_in_key_bin_index = 0;
                let mut min = f32::MAX;
                for j in 0..fft.bins() {
                    let diff = (nearest.0 - fft.frequency(j)).abs();
                    if diff < min {
                        min = diff;
                        nearest_in_key_bin_index = j;
                    }
                }
                // TODO:
                // parameterize

                let weight = nearest.1;
                let inverse_weight = 1.0 - weight;

                // log amplitude:
                // TODO:
                // use proper shelf-type filter to boost highs LOL
                let amp = (k as f32).log10() * 0.075;

                synthesis[k].1 += mag * amp * inverse_weight;

                vov[nearest_in_key_bin_index].0.push(mag);
                vov[nearest_in_key_bin_index].1 = weight;
            }
        }
        for (i, (v, weight)) in vov.iter().enumerate() {
            // TODO:
            // please rewrite this better.
            let sum = v.iter().sum::<f32>();
            let len = v.len();
            let avg = sum / max(len, 1) as f32;

            // TODO:
            // fix this if statement to prevent NaN LOLLL
            // parameterize this multiplier at the end!
            // figure out other stuff other than log, maybe parameterize that too!
            if i > 0 {
                let amplitude = (i as f32).log10() * 2.0 * weight;
                // println!("{}", amplitude);
                synthesis[i].1 += avg * amplitude;
            }
        }

        for k in 0..fft.bins() {
            let dry_signal = fft.at(channel, k);

            /* get magnitude and true frequency from synthesis array */
            let mag = synthesis[k].1;
            let mut freq = synthesis[k].0;
            /* subtract bin middle frequency */
            freq -= k as f32 * freq_per_bin;
            /* get bin deviation from freq deviation */
            freq /= freq_per_bin;
            /* take oversampling into account */
            freq = 2.0 * PI * freq / OVERSAMPLE_FACTOR as f32;
            /* add the overlap phase advance back in */
            freq += k as f32 * expected_phase;
            /* accumulate delta phase to get bin phase */
            phase_sum[k] += freq;
            let phase = phase_sum[k];
            /* get real and imag part.. */
            let real_component = mag * phase.sin();
            let imaginary_component = mag * phase.cos();
            let processed_signal = Complex32::new(real_component, imaginary_component);

            let current_freq = fft.frequency(k);

            let dry_wet_mix = processing_bandpass(10.0, 1_000.0, 100.0, 10_000.0, current_freq);
            let mixed = dry_wet_mix * processed_signal + (1.0 - dry_wet_mix) * dry_signal;

            let band_min = 20.0;
            let band_max = 44_100.0;

            if (band_min..=band_max).contains(&current_freq) {
                fft.set(channel, k, mixed);
            }
        }
    }
}
// TODO:
// add testing or make sure this works correctly LOLL
// https://www.desmos.com/calculator/paoj8fzugk

// takes an input frequency and returns some value between 0 and 1 bnased on if the frequency is in the band
fn processing_bandpass(
    hp_width: f32,
    lp_width: f32,
    hp_cutoff: f32,
    lp_cutoff: f32,
    x: f32,
) -> f32 {
    let a = 1.0;
    if (hp_cutoff > x && x > 0.0) || x > lp_cutoff {
        return 0.0;
    }
    if (lp_cutoff - lp_width) > x && x > (hp_cutoff + hp_width) {
        return a;
    }
    if (hp_cutoff + hp_width) > x && x > hp_cutoff {
        return (a * (x - hp_cutoff)) / hp_width;
    }
    if (lp_cutoff - lp_width) < x && x < lp_cutoff {
        return -(a * (x - lp_cutoff)) / lp_width;
    }

    0.0
}

fn is_in_key(in_key_frequencies: &[(f32, f32)], frequency: f32) -> bool {
    // would there be a way to optimize this?
    let mut result = false;
    for ikf in in_key_frequencies {
        // TODO:
        // tweak this tolerance!
        if (frequency - ikf.0).abs() <= 25.0 {
            result = true;
            break;
        }
    }
    result
}

// TODO:
// rewrite this
pub fn generate_frequencies(intervals: &[f32]) -> Vec<(f32, f32)> {
    // TODO:
    // turn this into a parameter of the function, being sample rate
    let sample_rate = 44_100.0;
    let max_frequency = sample_rate / 2.0; // Maximum frequency

    // convert our incoming array into an array of tuples
    let mut is = Vec::new();
    for (interval, velocity) in intervals.iter().enumerate() {
        is.push((interval, *velocity));
    }

    let mut freqs = Vec::new();

    // temporary variables
    let mut octave = 0;
    let mut base_note = 0;

    // TODO:
    // there is definitely some optimizations to be had here

    // also, good LORD please fix the indents
    while midi_to_freq(base_note as f32) <= max_frequency {
        for (interval, velocity) in &is {
            let midi_note = base_note + interval;
            if *velocity > 0.0 {
                let f = midi_to_freq(midi_note as f32);
                if f <= max_frequency {
                    // 2 is a good number for harmonics, at least in my opinion
                    // TODO:
                    // parameterize!

                    let num_harmonics = 2;
                    // note here that the 0th harmonic is the fundamental frequency
                    for nth_harmonic in 0..=num_harmonics {
                        // TODO:
                        // do something cool here
                        // let harmonic_weight = 1.0 - (nth_harmonic as f32 / num_harmonics as f32);
                        let harmonic_weight = 1.0;
                        let frequency = f * nth_harmonic as f32;

                        if frequency < max_frequency {
                            let tuple = (frequency, *velocity * harmonic_weight);
                            freqs.push(tuple);
                        }
                    }
                }
            }
        }
        octave += 1;
        base_note = octave * 12;
    }

    // TODO:
    // figure out how you want to sort??
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    freqs.dedup();

    freqs
}

fn midi_to_freq(x: f32) -> f32 {
    440.0 * 2.0_f32.powf((x - 69.0) / 12.0)
}

// i kept this here because uh.. idk i might use it later?
#[allow(dead_code)]
fn freq_to_midi(f: f32) -> f32 {
    12.0 * (f / 440.0).log2() + 69.0
}

mod tests {

    #[test]
    fn test_conversion() {
        use super::{freq_to_midi, midi_to_freq};
        assert_eq!(midi_to_freq(69.0), 440.0);
        assert_eq!(freq_to_midi(440.0), 69.0);
    }
}
