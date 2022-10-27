use crate::sparse::{definition::CxxMatrixf32, compress};



pub fn argmaxM(
    M: &CxxMatrixf32, 
    pi: &[i32], 
    row_block: i32, 
    col_block: i32
) -> CxxMatrixf32 {

    //println!("i: {:?}", M.i);
    //println!("j: {:?}", M.p);
    //println!("x: {:?}", M.x);
    let ridx = pi.iter()
        .enumerate()
        .map(|(i, x)| i as i32 + *x * row_block)
        .collect::<Vec<i32>>();
    let mut newM = CxxMatrixf32::new();
    let mut nz = 0;
    assert_eq!(row_block, pi.len() as i32);
    //println!("ridx: {:?}", ridx);
    for r in 0..row_block as usize {
        let rowlookup = ridx[r] as usize;
        let k = (M.i[rowlookup + 1] - M.i[rowlookup]) as usize;
        if k > 0 {
            for j_ in 0..k {
                if M.x[M.i[rowlookup] as usize + j_] != 0. {
                    newM.triple_entry(
                        r as i32, 
                        M.p[M.i[rowlookup] as usize + j_] - pi[r] * col_block, 
                        M.x[M.i[rowlookup] as usize + j_]
                    );
                    /*println!("M.P @ idx {} => Action: {} colblock: {}\n{:?}", 
                        M.i[rowlookup] as usize + j_, 
                        pi[r],
                        col_block,
                        M.p[M.i[rowlookup] as usize + j_]
                    );
                    println!("new triple: ({},{},{})", 
                        r, 
                        M.p[M.i[rowlookup] as usize + j_] - pi[r] * col_block,
                        M.x[M.i[rowlookup] as usize + j_]
                    );*/
                    nz += 1;
                }
            }
        }
    }

    //println!("cols from argmax func:\n{:?}", newM.p);

    newM.nz = nz;
    newM.m = row_block;
    newM.n = col_block;
    newM.nzmax = nz;

    
    println!("newM nnz: {}", newM.nz);
    println!("newM m: {}", newM.m);
    println!("newM n: {}", newM.n);
    /*
    println!("newMx:\n{:?}", newM.x);
    println!("newMi:\n{:?}", newM.i);
    println!("newMj:\n{:?}", newM.p);
    */
    newM = compress::compress(newM);
    newM
}