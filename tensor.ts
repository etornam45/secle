export type Shape = number[];

export class Tensor {
  data: Float32Array;
  shape: Shape;
  requires_grad: boolean;
  _grad?: Float32Array;
  _prev: Tensor[] = [];
  _name?: string;
  _backward?: () => void;

  constructor(
    data: Float32Array | number[],
    shape: Shape,
    requires_grad: boolean = false,
  ) {
    if (data instanceof Float32Array) {
      this.data = data;
    } else {
      this.data = new Float32Array(data);
    }
    this.shape = shape;
    this.requires_grad = requires_grad;
  }

  static fromArray(arr: number[], shape: Shape, requires_grad: boolean = false): Tensor {
    return new Tensor(arr, shape, requires_grad);
  }

  static zeros(shape: Shape, requires_grad: boolean = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size), shape, requires_grad);
  }

  static ones(shape: Shape, requires_grad: boolean = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).fill(1);
    return new Tensor(data, shape, requires_grad);
  }

  toString(): string {
    return `Tensor(shape=${this.shape}, data=[${Array.from(this.data).join(", ")}])`;
  }

  _shape(): Shape {
    return this.shape;
  }

  _requires_grad(): boolean {
    return this.requires_grad;
  }

  _data(): Float32Array {
    return this.data;
  }


  backward(grad?: Float32Array): void {
    if (!this.requires_grad) {
      throw new Error("Cannot call backward on a tensor that does not require gradients.");
    }
    if (!grad) {
      if (this.size() !== 1) throw new Error('grad must be provided for non-scalar tensors');
      grad = new Float32Array(this.data.length).fill(1);
    }
    this._grad = grad;

    // Build topo order
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    const buildTopo = (tensor: Tensor) => {
      if (!visited.has(tensor)) {
        visited.add(tensor);
      }
      for (const child of tensor._prev) {
        buildTopo(child);
      }
      topo.push(tensor);
    }

    buildTopo(this);
    topo.reverse();

    for (let i = 0; i < topo.length; i++) {
      const tensor = topo[i];
      if (tensor._backward) {
        tensor._backward();
      }
    }
  }

  size(): number {
    return this.data.length;
  }

  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error("Transpose is only implemented for 2D tensors.");
    }
    const [rows, cols] = this.shape;
    const result = Tensor.zeros([cols, rows], this.requires_grad);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result.data[j * rows + i] = this.data[i * cols + j];
      }
    }
    return result;
  }
}
