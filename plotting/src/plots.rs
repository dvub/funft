use plotters::prelude::*;
use std::path::Path;

pub fn frequency_plot(
    path: &Path,
    vecs: Vec<(Vec<f32>, RGBColor)>,
    frequencies: &Vec<f32>,
    _sample_rate: usize,
) {
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let max_y = vecs[0].0.clone().into_iter().reduce(f32::max).unwrap();
    let min_y = 0.0;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(":3", ("sans-serif", 40))
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d((1_000.0..10_000.0).log_scale(), (min_y..max_y).log_scale())
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

pub fn generic_plot(path: &Path, vecs: Vec<(Vec<f32>, RGBColor)>) {
    let root = BitMapBackend::new(&path, (1024, 768)).into_drawing_area();

    root.fill(&WHITE).unwrap();

    let max_y = vecs[0].0.clone().into_iter().reduce(f32::max).unwrap();
    let max_x = vecs[0].0.len() as f32;

    let min_y = vecs[0].0.clone().into_iter().reduce(f32::min).unwrap();
    let min_x = 0.0;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption(":3", ("sans-serif", 40))
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(min_x..max_x, -1.0..1.0_f32)
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(15)
        .y_labels(5)
        .x_desc("Sample #")
        .y_desc("Amp")
        .axis_desc_style(("sans-serif", 15))
        .draw()
        .unwrap();

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
