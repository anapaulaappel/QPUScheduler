OPENQASM 3.0;
bit[4] b;
#pragma braket verbatim
box {
  rz(1.5707963267948966) $3;
  rx(1.5707963267948966) $3;
  rz(-1.5707963267948966) $3;
}
b[0] = measure $0;
b[1] = measure $1;
b[2] = measure $2;
b[3] = measure $3;
