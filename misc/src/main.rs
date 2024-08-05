use hound::WavReader;
use plotters::prelude::*;
use realfft::RealFftPlanner;
use std::{
    f32::consts::PI,
    fs::create_dir,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

const OUT_DIR: &str = "./output";
fn main() {
    let (mut sine_samples, sine_sample_rate) = gen_piano_samples();

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(sine_samples.len());

    let mut output = r2c.make_output_vec();
    r2c.process(&mut sine_samples, &mut output).unwrap();

    let magnitude_height = |magnitude: f32| {
        let magnitude_db = gain_to_db(magnitude);
        (magnitude_db + 80.0) / 100.0
    };

    gen_chart(
        output
            .iter()
            .map(|complex| magnitude_height(complex.norm()))
            .collect(),
        sine_sample_rate as usize,
    );
}

pub fn gain_to_db(gain: f32) -> f32 {
    f32::max(gain, 1e-5).log10() * 20.0
}

fn gen_piano_samples() -> (Vec<f32>, f32) {
    let file = "c and g.wav";
    let mut reader = WavReader::open(Path::new(file)).unwrap();
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.bits_per_sample {
        16 => reader
            .samples::<i16>()
            .map(|s| {
                let sample_i16 = s.unwrap();
                sample_i16 as f32 / i16::MAX as f32
            })
            .collect(),
        _ => panic!("Unsupported bit depth"),
    };
    (samples, spec.sample_rate as f32)
}

fn gen_sine_waves() -> (Vec<f32>, f32) {
    let amplitude = 0.5;
    let frequency = 440.0; // Hz
    let frequency2 = frequency * 2.0;
    // https://www.gaussianwaves.com/2015/11/interpreting-fft-results-obtaining-magnitude-and-phase-information/
    let sample_rate = frequency2 * 2.0 * 32.0;
    let len = sample_rate * 2.0;

    let _resolution = sample_rate / len;

    let angular_freq = 2.0 * PI * frequency / sample_rate;
    let angular_freq2 = 2.0 * PI * frequency2 / sample_rate;

    let mut samples: Vec<f32> = vec![0.0; len as usize];

    for (i, v) in samples.iter_mut().enumerate() {
        let one = amplitude * (angular_freq * i as f32).sin();
        let two = amplitude * (angular_freq2 * i as f32).sin();
        *v = one + two;
    }
    (samples, sample_rate)
}

fn gen_chart(samples: Vec<f32>, sample_rate: usize) {
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

    let m = samples.clone().into_iter().reduce(f32::max).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(":3", ("sans-serif", 40))
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d((0..sample_rate).log_scale(), (0.0..m).log_scale())
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

    chart
        .draw_series(LineSeries::new(
            samples.iter().enumerate().map(|(i, x)| (i, *x)),
            BLUE.filled(),
        ))
        .unwrap();
    root.present().expect("Unable to write result to file.");
}
