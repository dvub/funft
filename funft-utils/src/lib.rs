use std::thread::current;

// TODO:
// rewrite ALL OF this dogshit code
// BECAUSE JESUS FUCK!
use fundsp::hacker::*;

pub fn process(fft: &mut FftWindow, frequencies: &[f32], depth: &Shared) {
    for channel in 0..=1 {
        for i in 0..fft.bins() {
            let current_frequency = fft.frequency(i);
            // for each bin's frequency, find the surrounding target frequencies
            if let Some((low_frequency, high_frequency)) =
                find_surrounding_frequencies(frequencies, current_frequency)
            {
                // find the middle between the 2 target frequencies
                let midpoint_frequency = (low_frequency + high_frequency) / 2.0;

                // get the distance to the midpoint, this is used for normalization
                let norm_factor = (midpoint_frequency - low_frequency).abs();
                // difference between the current frequency and midpoint
                // for example, if the current frequency is close to the midpoint, it will be quieter
                let diff = (current_frequency - midpoint_frequency).abs();
                // TODO
                // experiment with raising by a power
                let amp = (diff / norm_factor).powf(depth.value());

                let value = fft.at(channel, i);
                let adjusted_value = value * amp;

                // subtract
                fft.set(channel, i, fft.at(channel, i) * amp);
                // add
            }
        }
    }
}

fn midi_to_freq(x: f32) -> f32 {
    440.0 * 2.0_f32.powf((x - 69.0) / 12.0)
}

#[allow(dead_code)]
fn freq_to_midi(f: f32) -> f32 {
    12.0 * (f / 440.0).log2() + 69.0
}

pub fn generate_frequencies() -> Vec<f32> {
    let max_frequency = 44100.0; // Maximum frequency

    // C Major scale intervals (C, D, E, F, G, A, B, )
    let scale_intervals = vec![0, 2, 3, 5, 7, 8, 10];

    let mut freqs = Vec::new();
    let mut octave = 0;
    let mut base_note = 0;

    // Generate notes until the maximum frequency is reached
    while midi_to_freq(base_note as f32) <= max_frequency {
        for interval in &scale_intervals {
            let midi_note = base_note + interval;
            let f = midi_to_freq(midi_note as f32);
            if f <= max_frequency {
                freqs.push(f);
            }
        }
        octave += 1;
        base_note = octave * 12;
    }

    // Generate frequencies for each note in the scale
    freqs
}

pub fn find_surrounding_frequencies(v: &[f32], value: f32) -> Option<(f32, f32)> {
    // Ensure the vector is sorted and not empty
    /*if v.is_empty() || value < v[0] || value > v[v.len() - 1] {
        return None;
    }*/

    // Binary search to find the index where value would be inserted
    let mut low = 0;
    let mut high = v.len();

    while low < high {
        let mid = (low + high) / 2;
        if v[mid] < value {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    let idx = low;

    // Check if the value falls between adjacent elements
    if idx == 0 || idx == v.len() {
        return None; // Out of bounds
    }

    Some((v[idx - 1], v[idx]))
}

mod tests {

    #[test]
    fn test_conversion() {
        use super::{freq_to_midi, midi_to_freq};
        assert_eq!(midi_to_freq(69.0), 440.0);
        assert_eq!(freq_to_midi(440.0), 69.0);
    }
    #[test]
    fn test_generate_frequencies() {
        use super::generate_frequencies;
        println!("{:?}", generate_frequencies());
    }
    #[test]
    fn test_adjacent() {
        let vec = vec![0.5, 1.0, 3.0, 5.0];
        let v1 = 2.0;
        let v2 = 0.7;
        assert_eq!(
            super::find_surrounding_frequencies(&vec, v1).unwrap(),
            (1.0, 3.0)
        );
        assert_eq!(
            super::find_surrounding_frequencies(&vec, v2).unwrap(),
            (0.5, 1.0)
        );
    }
}
