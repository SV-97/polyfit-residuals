use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use polyfit_residuals::{all_residuals, all_residuals_par, residuals_from_front};

fn run_residuals_from_front(data_len: usize, max_degree: usize) {
    let xs = Array1::linspace(1., 10_000_000., data_len);
    let ys: Array1<f64> = xs.clone() * xs.clone() + 2. * xs.clone() - 1.;
    residuals_from_front(
        black_box(xs.view()),
        black_box(ys.view()),
        black_box(max_degree),
    )
    .unwrap();
}

fn run_all_residuals(data_len: usize, max_degree: usize) {
    let xs = Array1::linspace(1., 10_000_000., data_len);
    let ys: Array1<f64> = xs.clone() * xs.clone() + 2. * xs.clone() - 1.;
    all_residuals(
        black_box(xs.view()),
        black_box(ys.view()),
        black_box(max_degree),
    );
}

fn run_all_residuals_par(data_len: usize, max_degree: usize) {
    let xs = Array1::linspace(1., 10_000_000., data_len);
    let ys: Array1<f64> = xs.clone() * xs.clone() + 2. * xs.clone() - 1.;
    all_residuals_par(
        black_box(xs.view()),
        black_box(ys.view()),
        black_box(max_degree),
    );
}

pub fn criterion_benchmark(c: &mut Criterion) {
    for data_len in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] {
        let deg = 6;
        c.bench_function(
            &format!("small_data / residuals_from_front / data_len={data_len}, deg<={deg}"),
            |b| b.iter(|| run_residuals_from_front(data_len, 6)),
        );
        c.bench_function(
            &format!("small_data / all_residuals / data_len={data_len}, deg<={deg}"),
            |b| b.iter(|| run_all_residuals(data_len, deg)),
        );
        c.bench_function(
            &format!(
                "small_data / all_residuals_par / data_len={}, deg<={deg}",
                data_len
            ),
            |b| b.iter(|| run_all_residuals_par(data_len, deg)),
        );
    }

    for (data_len, deg) in [(100, 10), (500, 10), (1_000, 10), (5_000, 10), (1_000, 100)] {
        c.bench_function(
            &format!("large_data / residuals_from_front / data_len={data_len}, deg<={deg}"),
            |b| b.iter(|| run_residuals_from_front(data_len, deg)),
        );
        c.bench_function(
            &format!("large_data / all_residuals / data_len={data_len}, deg<={deg}"),
            |b| b.iter(|| run_all_residuals(data_len, deg)),
        );
        c.bench_function(
            &format!(
                "large_data / all_residuals_par / data_len={}, deg<={deg}",
                data_len
            ),
            |b| b.iter(|| run_all_residuals_par(data_len, deg)),
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
