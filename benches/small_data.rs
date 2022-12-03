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
    for data_len in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] {
        c.bench_function(
            &format!("residuals_from_front n={}, deg<=6", data_len),
            |b| b.iter(|| run_residuals_from_front(data_len, 6)),
        );
    }

    for data_len in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] {
        c.bench_function(&format!("all_residuals n={}, deg<=6", data_len), |b| {
            b.iter(|| run_all_residuals(data_len, 6))
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
