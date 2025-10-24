## <img src="./assets/sircle.png" width="30" style="margin: 0;" /> Secle - A simple ml framework built for `Typescript`



Secle takes inspiration from pytorch's API. 

> NOTE: The library supports webgpu but it is not optimized yet. 
> Secle can work with deno 2.1 ^ with the unstable flags for webgpu. 
> You can also run secle in jupyter notebook with deno


## To use Secle

```ts
import * as nn from "https://raw.githubusercontent.com/etornam45/torch-ts/refs/heads/main/dist/bundle.ts"
```

At the moment the library is not available on npm or jsr. So, you can only use if like this - Which means there will always be changes to the API until it's Available on npm or jsr

## A simple Network 

```ts

class Model extends nn.Module {
  seq: nn.Sequencial;
  
  constructor(input_size: number, output_size: number, hidden_dim = 10) {
    super();
    this.seq = new nn.Sequencial([
      new nn.Linear(input_size, hidden_dim),
      new nn.Sigmoid(),
      new nn.Linear(hidden_dim, hidden_dim),
      new nn.Sigmoid(),
      new nn.Linear(hidden_dim, hidden_dim),
      new nn.Sigmoid(),
      new nn.Linear(hidden_dim, output_size)
    ])
    this.register_parameters([this.seq]);
  }

  override forward(x: nn.Tensor): nn.Tensor {
    return this.seq.$(x); // The .$(x) is just a wrapper around .forward(x)
  }
}

```

This is a simple Neural Network with `2 hidden` layers

```ts
// Let's train the model

const device: nn.Device = "cpu" // webgpu or cpu
const model = new Model(1, 1, 1000);
await model.to(device)
const optimizer = new SGD(model.parameters, 0.000001);
const criterion = nn.MSELoss()
```


```ts
for (let i = 0; i < 10; i++) {
  for (const { x, y } of data) { // Asuming your data is {x : Tensor, y: Tensor }
    await x.to(device)
    await y.to(device)
    const output = model.$(x);
    const loss = criterion(output, y);
    loss.backward(); // Calculate the gradients
    optimizer.step(); // Learn
    optimizer.zero_grad(); // Zero_grad
  }
}
```

> I want to improve the training api to make it more simpler like

```ts
model.train(...)
```

