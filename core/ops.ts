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


export function mul(a: Tensor | number, b: Tensor | number): Tensor {

  if (typeof a === "number" && b instanceof Tensor) {
    return CPU.mul_scalar(b, a);
  }
  if (typeof b === "number" && a instanceof Tensor) {
    return CPU.mul_scalar(a, b);
  }

  if (a instanceof Tensor && b instanceof Tensor) {
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
  // if both are numbers, return a new tensor with the product
  if (typeof a === "number" && typeof b === "number") {
    return new Tensor([a * b], [1, 1]);
  }

  throw new Error("Invalid arguments to mul. Expected a Tensor or a number, got " + typeof a + " and " + typeof b);
}

export function add(a: Tensor | number, b: Tensor | number): Tensor {
  if (typeof a === "number" && b instanceof Tensor) {
    return CPU.add_a_number(b, a);
  }
  if (typeof b === "number" && a instanceof Tensor) {
    return CPU.add_a_number(a, b);
  }

  if (a instanceof Tensor && b instanceof Tensor) {
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

  if (typeof a === "number" && typeof b === "number") {
    return new Tensor([a + b], [1, 1]);
  }

  throw new Error("Invalid arguments to add. Expected a Tensor or a number, got " + typeof a + " and " + typeof b);
}


export function mean(a: Tensor): Tensor {
  const out = CPU.mean(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      // d mean / d a_i = 1/N
      const scale = 1 / a.size();
      const scaled = mul(out._grad, scale);
      const gradBroadcast = new Tensor(
        new Float32Array(a.size()).fill(scaled.data[0]),
        a.shape,
        a.requires_grad,
      );
      a._grad = a._grad ? add(a._grad, gradBroadcast) : gradBroadcast;
    }
  };
  return out;
}

export function pow(a: Tensor, b: number): Tensor {
  const out = CPU.pow(a, b);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      // d (a^b) / d a = b * a^(b-1)
      const local = mul(CPU.pow(a, b - 1), b);
      const grad = mul(out._grad, local);
      a._grad = a._grad ? add(a._grad, grad) : grad;
    }
  };
  return out;
}

export function sum(a: Tensor): Tensor {
  const out = CPU.sum(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      // d sum / d a_i = 1
      const gradBroadcast = new Tensor(
        new Float32Array(a.size()).fill(out._grad.data[0]),
        a.shape,
        a.requires_grad,
      );
      a._grad = a._grad ? add(a._grad, gradBroadcast) : gradBroadcast;
    }
  };
  return out;
}


export function relu(a: Tensor): Tensor {
  const out = CPU.relu(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      a._grad = a._grad ? add(a._grad, relu(out._grad)) : relu(out._grad);
    }
  };
  return out;
}

export function sigmoid(a: Tensor): Tensor {
  const out = CPU.sigmoid(a);
  out._prev = [a];
  out.requires_grad = a.requires_grad;
  out._backward = () => {
    if (!out._grad) return;
    if (a.requires_grad) {
      a._grad = a._grad ? add(a._grad, sigmoid(out._grad)) : sigmoid(out._grad);
    }
  };
  return out;
}