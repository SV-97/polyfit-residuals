use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use polyfit_residuals::{all_residuals, residuals_from_front};

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
pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("all_residuals 100, deg<=10", |b| {
        b.iter(|| run_all_residuals(100, 10))
    });

    c.bench_function("all_residuals 500, deg<=10", |b| {
        b.iter(|| run_all_residuals(500, 10))
    });

    c.bench_function("all_residuals 1k, deg<=10", |b| {
        b.iter(|| run_all_residuals(1_000, 10))
    });

    c.bench_function("all_residuals 5k, deg<=10", |b| {
        b.iter(|| run_all_residuals(5_000, 10))
    });

    c.bench_function("all_residuals 1k, deg<=100", |b| {
        b.iter(|| run_all_residuals(1_000, 100))
    });

    c.bench_function("residuals_from_front 50k, deg<=2", |b| {
        b.iter(|| run_residuals_from_front(50_000, 2))
    });

    c.bench_function("residuals_from_front 50k, deg<=10", |b| {
        b.iter(|| run_residuals_from_front(50_000, 10))
    });

    c.bench_function("residuals_from_front 5k, deg<=100", |b| {
        b.iter(|| run_residuals_from_front(5_000, 100))
    });

    c.bench_function("residuals_from_front 5k, deg<=1000", |b| {
        b.iter(|| run_residuals_from_front(5_000, 1_000))
    });
    // let mut large_group = c.benchmark_group("large-cases");
    // large_group.sampling_mode(SamplingMode::Flat);
    // large_group.sample_size(10);
    // large_group.bench_function("residuals_from_front 50k, deg<=10k", |b| {
    //     b.iter(|| run_residuals_from_front(50_000, 10_000))
    // });
    // large_group.bench_function("residuals_from_front 1M, deg<=10k", |b| {
    //     b.iter(|| run_residuals_from_front(1_000_000, 10_000))
    // });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
