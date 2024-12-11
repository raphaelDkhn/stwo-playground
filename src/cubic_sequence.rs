use itertools::Itertools;
use num_traits::Zero;
use stwo_prover::{
    constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval, TraceLocationAllocator},
    core::{
        air::Component,
        backend::{
            simd::{
                m31::{PackedBaseField, LOG_N_LANES},
                SimdBackend,
            },
            BackendForChannel, Col, Column,
        },
        channel::MerkleChannel,
        fields::{m31::BaseField, qm31::SecureField, FieldExpOps},
        pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig},
        poly::{
            circle::{CanonicCoset, CircleEvaluation, PolyOps},
            BitReversedOrder,
        },
        prover::{prove, verify, StarkProof, VerificationError},
        ColumnVec,
    },
};

type WorkComponent<const N: usize> = FrameworkComponent<WorkEval<N>>;

struct WorkInput {
    start: PackedBaseField,
}

/// A component that enforces a sequence where each element is cubed and summed with 42.
/// Each row contains a separate sequence of length `N`.
#[derive(Clone)]
pub struct WorkEval<const N: usize> {
    pub log_n_rows: u32,
}

impl<const N: usize> FrameworkEval for WorkEval<N> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut prev = eval.next_trace_mask();
        for _ in 1..N {
            let curr: <E as EvalAtRow>::F = eval.next_trace_mask();
            let forty_two: <E as EvalAtRow>::F = E::F::from(BaseField::from_u32_unchecked(42));
            // Constraint: curr = prev^3 + 42
            eval.add_constraint(curr.clone() - (prev.pow(3) + forty_two));
            prev = curr;
        }
        eval
    }
}

fn generate_trace<const N: usize>(
    log_size: u32,
    inputs: &[WorkInput],
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    // Initialize the trace matrix with N columns (steps in sequence)
    // Each column has 2^log_size rows (number of parallel sequences)
    let mut trace = (0..N)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();

    // Create a SIMD-packed constant value of 42 that will be used in each computation
    let forty_two = PackedBaseField::broadcast(BaseField::from_u32_unchecked(42));

    // Iterate through chunks of 16 sequences (due to AVX-512 SIMD register size)
    // vec_index represents which chunk of 16 sequences we're processing
    for (vec_index, input) in inputs.iter().enumerate() {
        // Get the starting values for this chunk of 16 sequences
        // current is a PackedBaseField containing 16 field elements
        let mut current = input.start;

        // Store the 16 starting values in the first column
        trace[0].data[vec_index] = current;

        // For each subsequent column (step in the sequences)
        // Skip the first column since we already filled it
        trace.iter_mut().skip(1).for_each(|col| {
            // Compute the next values for all 16 sequences simultaneously using SIMD:
            // 1. Cube all 16 values (current.pow(3))
            // 2. Add 42 to all 16 values using SIMD addition
            current = current.pow(3) + forty_two;

            // Store the 16 computed values in the current column
            col.data[vec_index] = current;
        });
    }

    // Create a circle domain of size 2^log_size
    // This domain is used for the Circle STARK proof system
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Transform each column into a CircleEvaluation
    trace
        .into_iter()
        .map(|eval| CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(domain, eval))
        .collect_vec()
}

const SEQUENCE_LENGTH: usize = 100;

fn prove_sequences<MC: MerkleChannel>(
    log_size: u32,
    config: PcsConfig,
) -> (WorkComponent<SEQUENCE_LENGTH>, StarkProof<MC::H>)
where
    SimdBackend: BackendForChannel<MC>,
{
    // Precompute twiddles for FFT operations
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Setup protocol
    let prover_channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<SimdBackend, MC>::new(config, &twiddles);

    // Generate inputs based on instance count
    let inputs = if log_size < LOG_N_LANES {
        // For small instance counts, pack them into a single SIMD vector
        let n_instances = 1 << log_size;
        vec![WorkInput {
            start: PackedBaseField::from_array(std::array::from_fn(|j| {
                if j < n_instances {
                    BaseField::from_u32_unchecked(j as u32)
                } else {
                    BaseField::zero()
                }
            })),
        }]
    } else {
        // For normal case, use full SIMD width
        (0..(1 << (log_size - LOG_N_LANES)))
            .map(|i| WorkInput {
                start: PackedBaseField::from_array(std::array::from_fn(|j| {
                    BaseField::from_u32_unchecked((i * 16 + j) as u32)
                })),
            })
            .collect_vec()
    };

    // Preprocess Trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([]);
    tree_builder.commit(prover_channel);

    // Generate Trace
    let trace = generate_trace::<SEQUENCE_LENGTH>(log_size, &inputs);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);

    // Create Component
    let component = WorkComponent::new(
        &mut TraceLocationAllocator::default(),
        WorkEval::<SEQUENCE_LENGTH> {
            log_n_rows: log_size,
        },
        (SecureField::zero(), None),
    );

    // Generate Proof
    let proof = prove::<SimdBackend, MC>(&[&component], prover_channel, commitment_scheme).unwrap();

    (component, proof)
}

fn verify_sequences<MC: MerkleChannel>(
    config: PcsConfig,
    component: WorkComponent<SEQUENCE_LENGTH>,
    proof: StarkProof<MC::H>,
) -> Result<(), VerificationError> {
    let channel = &mut MC::C::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<MC>::new(config);

    // Get expected column sizes from component
    let log_sizes = component.trace_log_degree_bounds();

    // Verify main trace commitment
    commitment_scheme.commit(proof.commitments[0], &log_sizes[0], channel);

    // Verify constant trace commitment
    commitment_scheme.commit(proof.commitments[1], &log_sizes[1], channel);

    // Verify the proof
    verify(&[&component], channel, commitment_scheme, proof)
}

#[cfg(test)]
mod tests {
    use stwo_prover::core::{pcs::PcsConfig, vcs::blake2_merkle::Blake2sMerkleChannel};

    use super::*;

    #[test]
    fn test_small_instance_count() {
        // Test specifically with small number of instances (less than SIMD width)
        let log_n_instances = 2; // 4 sequences
        let config = PcsConfig::default();

        let (component, proof) = prove_sequences::<Blake2sMerkleChannel>(log_n_instances, config);

        verify_sequences::<Blake2sMerkleChannel>(config, component, proof).unwrap();
    }

    #[test]
    fn test_max_instance_count() {
        // Test with maximum practical instance count
        let log_n_instances = 6; // 64 sequences
        let config = PcsConfig::default();

        let (component, proof) = prove_sequences::<Blake2sMerkleChannel>(log_n_instances, config);

        verify_sequences::<Blake2sMerkleChannel>(config, component, proof).unwrap();
    }
}
