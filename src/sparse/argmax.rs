use crate::sparse::{definition::CxxMatrixf32, compress};

/// argmax on a compressed row matrix
pub fn argmaxM(
    M: &CxxMatrixf32, 
    pi: &[i32], 
    row_block: i32, 
    col_block: i32
) -> CxxMatrixf32 {
    let ridx = pi.iter()
        .enumerate()
        .map(|(i, x)| i as i32 + *x * row_block)
        .collect::<Vec<i32>>();
    let mut newM = CxxMatrixf32::new();
    let mut nz = 0;

    assert_eq!(row_block, pi.len() as i32);

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
                    nz += 1;
                }
            }
        }
    }

    newM.nz = nz;
    newM.m = row_block;
    newM.n = col_block;
    newM.nzmax = nz;

    newM = compress::compress(newM);
    newM
}

pub fn multiobj_argmaxP(
    M: &CxxMatrixf32, 
    pi: &[i32], 
    row_block: i32, 
    col_block: i32,
    nobjs: usize
) -> CxxMatrixf32 {
    let ridx = pi.iter()
        .enumerate()
        .map(|(i, x)| i as i32 + *x * row_block)
        .collect::<Vec<i32>>();
    let mut newM = CxxMatrixf32::new();
    let mut nz = 0;

    assert_eq!(row_block, pi.len() as i32);

    for obj in 0..nobjs {
        for r in 0..row_block as usize {
            let rowlookup = ridx[r] as usize;
            let k = (M.i[rowlookup + 1] - M.i[rowlookup]) as usize;
            if k > 0 {
                for j_ in 0..k {
                    if M.x[M.i[rowlookup] as usize + j_] != 0. {
                        newM.triple_entry(
                            obj as i32 * row_block + r as i32, 
                            obj as i32 * col_block + M.p[M.i[rowlookup] as usize + j_] - pi[r] * col_block, 
                            M.x[M.i[rowlookup] as usize + j_]
                        );
                        nz += 1;
                    }
                }
            }
        }
    }

    newM.nz = nz;
    newM.m = row_block * nobjs as i32;
    newM.n = col_block * nobjs as i32;
    newM.nzmax = nz;

    newM = compress::compress(newM);
    newM
}


pub fn multiobj_argmaxR(
    M: &CxxMatrixf32, 
    pi: &[i32], 
    row_block: i32, 
    nobjs: usize
) -> Vec<f32> {
    let mut output: Vec<f32> = vec![0.; nobjs * row_block as usize];
    let ridx = pi.iter()
        .enumerate()
        .map(|(i, x)| i as i32 + *x * row_block)
        .collect::<Vec<i32>>();

    assert_eq!(row_block, pi.len() as i32);

    for r in 0..row_block as usize {
        let rowlookup = ridx[r] as usize;
        let k = (M.i[rowlookup + 1] - M.i[rowlookup]) as usize;
        if k > 0 {
            for j_ in 0..k {
                if M.x[M.i[rowlookup] as usize + j_] != 0. {
                    //println!("pi[r]: {}", pi[r]);
                    //println!("p: {}, rowblock: {}", M.p[M.i[rowlookup] as usize + j_], nobjs);
                    let c: usize = M.p[M.i[rowlookup] as usize + j_] as usize 
                        - pi[r] as usize * nobjs; 
                    let idx = (c % nobjs) * row_block as usize + r;
                    //println!("idx {} = [{} = {}] * {} + {}", idx, c, c % 4, row_block, r);
                    output[idx] = M.x[M.i[rowlookup] as usize + j_];
                }
            }
        }
    }
    output
}