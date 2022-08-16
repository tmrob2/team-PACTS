use ce;

#[test]
fn dense_matrix_contruction() -> Result<(), Box<dyn std::error::Error>> {
    ce::DenseMatrix::new(0, 0);
    Ok(())
}