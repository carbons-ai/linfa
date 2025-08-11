mod algorithm;
mod errors;
mod hyperparams;

pub use algorithm::*;
pub use errors::*;
pub use hyperparams::*;

#[cfg(test)]
mod tests {

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    //use super::*;
    
    use ndarray::array;

    #[allow(unused_variables)]
    #[test]
    fn test_gmm() {

/**
 * 
fn estimate_gaussian_covariances_diag(
    resp: &Array2<f64>,
    x: &Array2<f64>,
    nk: &Array1<f64>,
    means: &Array2<f64>,
    reg_covar: f64,
) -> Array2<f64> {
    // avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    let x2 = x.mapv(|v| v * v);
    let avg_x2 = resp.t().dot(&x2) / &nk.insert_axis(Axis(1));
    // avg_means2 = means**2
    let avg_means2 = means.mapv(|v| v * v);
    // return avg_X2 - avg_means2 + reg_covar
    &avg_x2 - &avg_means2 + reg_covar
}
 * 
 */

        let ndata = 6;
        let nfeatures = 2;
        let nclusters = 2;

        let nk = 2.0;
        let x = array![[1.0, 2.0], [1.0, 4.0], [1.0, 0.0], [10.0, 2.0], [10.0, 4.0], [10.0, 0.0]];
        let resp = array![[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4]];
        let squared = x.mapv(|x| x * x) / nk;

        print!("{:?}", squared);
    }
}
