fn main() {
    env_logger::init();
    // eprintln!("{}", dirichlet_fit::digam_inv(1f64));
    // eprintln!("{}", dirichlet_fit::digam_inv(-1f64));
    // eprintln!("{}", dirichlet_fit::digam_inv(-10f64));
    // eprintln!("{}", dirichlet_fit::digam_inv(-100f64));
    // eprintln!("{}", dirichlet_fit::digam_inv(-1000f64));
    // eprintln!("{}", dirichlet_fit::digam_inv(-2000f64));
    let sums = [-0.13839969047131848, -4.906374686615713];
    // let sums = [
    //     -31.608824312984375,
    //     -0.0006407897850113642,
    //     -31.57495419708601,
    //     -7.353129481056954,
    //     -31.6094379124341,
    // ];
    let mut estim = estim(&sums);
    let mut buf = vec![0f64; estim.len()];
    dirichlet_fit::fit_dirichlet_only_mean(&sums, &mut estim, &mut buf);
    eprintln!("LOSS\t{:?}\t{:?}", sums, estim);
}

fn estim(xs: &[f64]) -> Vec<f64> {
    let mut param: Vec<_> = xs
        .iter()
        .map(|&x| {
            if -2.2 < x {
                x.exp() + 0.5
            } else {
                -(x - dirichlet_fit::DIGAM_ONE).recip()
            }
        })
        .collect();
    let sum: f64 = param.iter().sum();
    let norm = 2f64;
    param.iter_mut().for_each(|x| *x = *x * norm / sum);
    param
}
