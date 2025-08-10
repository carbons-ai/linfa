use linfa_clustering::GaussianMixtureModel;
use linfa::traits::Fit;
use linfa::traits::Predict;
use linfa::DatasetBase;
use linfa_datasets::generate;
use ndarray::{array, Axis};
use ndarray_npy::write_npy;
use ndarray_rand::rand::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use linfa_nn::distance::LInfDist;

// A GMM task: build a synthetic dataset, fit the algorithm on it
// and save both training data and predictions to disk.
fn main() {
    // Our random number generator, seeded for reproducibility
    let mut rng = Xoshiro256Plus::seed_from_u64(42);

    // For each our expected centroids, generate `n` data points around it (a "blob")
    let expected_centroids = array![[10., 10.], [1., 12.], [20., 30.], [-20., 30.],];
    let n = 10000;
    let dataset = DatasetBase::from(generate::blobs(n, &expected_centroids, &mut rng));

    // Configure our training algorithm
    let n_clusters = expected_centroids.len_of(Axis(0));
    let gmm = GaussianMixtureModel::params(n_clusters)
        .n_runs(10)
        .tolerance(1e-4)
        .with_rng(rng)
        .fit(&dataset).expect("GMM fitting");

    // Assign each point to a cluster using the set of centroids found using `fit`
    let dataset = gmm.predict(dataset);
    let DatasetBase {
        records, targets, ..
    } = dataset;

    // Save to disk our dataset (and the cluster label assigned to each observation)
    // We use the `npy` format for compatibility with NumPy
    write_npy("clustered_dataset.npy", &records).expect("Failed to write .npy file");
    write_npy("clustered_memberships.npy", &targets.map(|&x| x as u64)).expect("Failed to write .npy file");

    println!("GMM means = {:?}", gmm.means());
    println!("GMM covariances = {:?}", gmm.covariances());
}
