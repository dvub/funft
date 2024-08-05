use fundsp::hacker::*;

use funft_utils::{generate_frequencies, process};

use numeric_array::{generic_array::arr, NumericArray};
use plotters::prelude::*;
use rand::Rng;
use realfft::RealFftPlanner;

use std::{
    fs::create_dir,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};
const OUT_DIR: &str = "./output";

fn main() {
    let window_length = 1024;
    let frequencies = generate_frequencies();

    let mut noise = gen_noise();
    // save an unprocessed copy
    let mut pre_noise = noise.clone();
    let v = shared(1.5);
    let mut synth = resynth::<U2, U2, _>(window_length, |fft| {
        process(fft, &frequencies, &v);
    });
    for sample in &mut noise {
        let samples_into_array = &NumericArray::new(arr![*sample; 2]);
        *sample = synth.tick(samples_into_array)[0];
    }

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(pre_noise.len());

    let mut output = r2c.make_output_vec();
    r2c.process(&mut pre_noise, &mut output).unwrap();

    let mut output2 = r2c.make_output_vec();
    r2c.process(&mut noise, &mut output2).unwrap();

    let noise_fft_mags: Vec<_> = output.iter().map(|s| s.norm()).collect();
    let processed_fft_mags: Vec<_> = output2.iter().map(|s| s.norm()).collect();
    gen_chart(
        vec![(noise_fft_mags, BLUE), (processed_fft_mags, RED)],
        &frequencies,
        44100,
    );
}

pub fn gain_to_db(gain: f32) -> f32 {
    f32::max(gain, 1e-5).log10() * 20.0
}

fn gen_noise() -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut v = Vec::new();
    let amplitude = 0.25;
    for _ in 0..=44100 {
        v.push(rng.gen_range(-amplitude..amplitude));
    }
    v
}

fn gen_chart(vecs: Vec<(Vec<f32>, RGBColor)>, frequencies: &Vec<f32>, _sample_rate: usize) {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();

    if !Path::new(OUT_DIR).exists() {
        create_dir(OUT_DIR).unwrap();
    }

    let path = Path::new(OUT_DIR).join(format!("{}.png", time));
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let max_y = vecs[0].0.clone().into_iter().reduce(f32::max).unwrap();
    let min_y = 0.0;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(":3", ("sans-serif", 40))
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d((5000.0..6000.0).log_scale(), (min_y..max_y).log_scale())
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(15)
        .y_labels(5)
        .x_desc("Frequency (Hz)")
        .y_desc("FFT Magnitude")
        .axis_desc_style(("sans-serif", 15))
        .draw()
        .unwrap();

    for frequency in frequencies {
        let v = vec![(*frequency, min_y), (*frequency, max_y)];
        chart
            .draw_series(LineSeries::new(v.into_iter(), GREEN.stroke_width(3)))
            .unwrap();
    }

    for vec in vecs {
        chart
            .draw_series(LineSeries::new(
                vec.0.iter().enumerate().map(|(i, x)| (i as f32, *x)),
                vec.1.filled(),
            ))
            .unwrap();
    }

    root.present().expect("Unable to write result to file.");
}
