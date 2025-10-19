import { Tensor } from "./tensor.ts";
import { CPU } from "./backends/cpu.ts";

export function matmul(a: Tensor, b: Tensor): Tensor {
  const out = CPU.matmul(a, b);
  out._prev = [a, b];
  out.requires_grad = a.requires_grad || b.requires_grad; // Set requires_grad based on inputs

  out._backward = () => {
    if (!out._grad) return;
    
    if (a.requires_grad) {
      const a_grad = matmul(out._grad, b.transpose());
      a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
    }
    
    if (b.requires_grad) {
      const b_grad = matmul(a.transpose(), out._grad);
      b._grad = b._grad ? add(b._grad, b_grad) : b_grad;
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
      a._grad = a._grad ? add(a._grad, out._grad) : out._grad;
    }
    if (b.requires_grad) {
      const negGrad = neg(out._grad);
      b._grad = b._grad ? add(b._grad, negGrad) : negGrad;
    }
  };

  return out;
}

export function neg(a: Tensor): Tensor {
  const out = CPU.neg(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      a._grad = a._grad ? add(a._grad, neg(out._grad)) : neg(out._grad);
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
      const a_grad = mul(a, out._grad);
      a._grad = a._grad ? add(a._grad, a_grad) : a_grad;
    }
    if (b.requires_grad) {
      const b_grad = mul(out._grad, a);
      b._grad = b._grad ? add(b._grad, b_grad) : b_grad;
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
      a._grad = a._grad ? add(a._grad, out._grad) : out._grad;
    }
    if (b.requires_grad) {
      b._grad = b._grad ? add(b._grad, out._grad) : out._grad;
    }
  };

  return out;
}
