import { Tensor } from "./tensor.ts";
import { CPU } from "./cpu.ts";

export function matmul(a: Tensor, b: Tensor): Tensor {
  const out = CPU.matmul(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad; // Set requires_grad based on inputs

  out._backward = () => {
    if (!out._grad) return;
    const [m, n] = out.shape;
    const [, k] = a.shape;
    
    if (a.requires_grad) {
      const a_grad = Tensor.zeros(a.shape);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < k; j++) {
          let sum = 0;
          for (let p = 0; p < n; p++) {
            sum += out._grad[i * n + p] * b.data[j * n + p];
          }
          a_grad.data[i * k + j] = sum;
        }
      }
      a._grad = a._grad ? addArrays(a._grad, a_grad.data) : a_grad.data;
    }
    
    if (b.requires_grad) {
      const b_grad = Tensor.zeros(b.shape);
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < n; j++) {
          let sum = 0;
          for (let p = 0; p < m; p++) {
            sum += out._grad[p * n + j] * a.data[p * k + i];
          }
          b_grad.data[i * n + j] = sum;
        }
      }
      b._grad = b._grad ? addArrays(b._grad, b_grad.data) : b_grad.data;
    }
  };

  return out;
}

export function sub(a: Tensor, b: Tensor): Tensor {
  const out = CPU.sub(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad;

  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      a._grad = a._grad ? addArrays(a._grad, out._grad) : out._grad;
    }
    if (b.requires_grad) {
      const negGrad = new Float32Array(out._grad.length);
      for (let i = 0; i < out._grad.length; i++) {
        negGrad[i] = -out._grad[i];
      }
      b._grad = b._grad ? addArrays(b._grad, negGrad) : negGrad;
    }
  };

  return out;
}


export function mul(a: Tensor, b: Tensor): Tensor {
  const out = CPU.mul(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad;
  
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      const a_grad = new Float32Array(a.data.length);
      for (let i = 0; i < a.data.length; i++) {
        a_grad[i] = b.data[i] * out._grad[i];
      }
      a._grad = a._grad ? addArrays(a._grad, a_grad) : a_grad;
    }
    if (b.requires_grad) {
      const b_grad = new Float32Array(b.data.length);
      for (let i = 0; i < b.data.length; i++) {
        b_grad[i] = a.data[i] * out._grad[i];
      }
      b._grad = b._grad ? addArrays(b._grad, b_grad) : b_grad;
    }
  };
  
  return out;
}

export function add(a: Tensor, b: Tensor): Tensor {
  const out = CPU.add(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad;

  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      a._grad = a._grad ? addArrays(a._grad, out._grad) : out._grad;
    }
    if (b.requires_grad) {
      b._grad = b._grad ? addArrays(b._grad, out._grad) : out._grad;
    }
  };

  return out;
}

function addArrays(a: Float32Array, b: Float32Array) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}
