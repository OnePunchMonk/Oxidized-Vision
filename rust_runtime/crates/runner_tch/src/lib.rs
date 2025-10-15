use tch::{CModule, Tensor, Device};
use ndarray::Array4;

pub struct TchRunner {
    module: CModule,
    device: Device,
}

impl TchRunner {
    pub fn load(path: &str, use_cuda: bool) -> anyhow::Result<Self> {
        let device = if use_cuda && tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
        let module = CModule::load_on_device(path, device)?;
        Ok(Self { module, device })
    }

    pub fn run(&self, input: Array4<f32>) -> anyhow::Result<Array4<f32>> {
        let input_shape = input.shape();
        let shape: Vec<i64> = input_shape.iter().map(|&d| d as i64).collect();
        let t = Tensor::from(input.as_slice().unwrap()).view(shape.as_slice()).to_device(self.device);
        let out = self.module.forward_ts(&[t])?;
        let out = out.to_device(Device::Cpu);
        let out_vec: Vec<f32> = out.try_into()?;
        let out_shape = out.size().iter().map(|&d| d as usize).collect::<Vec<_>>();
        
        // This is a simplification. You might need to handle different output shapes.
        if out_shape.len() == 4 {
            Ok(Array4::from_shape_vec((out_shape[0], out_shape[1], out_shape[2], out_shape[3]), out_vec)?)
        } else {
            // Handle other dimensions or return an error
            anyhow::bail!("Unsupported output dimension")
        }
    }
}
