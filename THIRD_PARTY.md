# Third-party provenance

- RUA / `hebbian-rua` 0.1.2: MIT; runtime dependency, installed separately. Its SO-101 geometry is read from the public wheel at runtime.
- SO-101 geometry: TheRobotStudio/SO-ARM100, Apache-2.0 as packaged by RUA. No geometry files are included here; generated model XML and meshes stay in ignored run directories.
- MuJoCo: Apache-2.0; MLX: MIT; NumPy: BSD-3-Clause; SciPy: BSD-3-Clause; Pillow: HPND; OpenCV: Apache-2.0. Installations include their own license notices.
- The original broader experiment was initially motivated by MacMimic, https://github.com/air23zj/macmimic. This visual pickup baseline is behavior cloning with simulation DAgger, not a reproduction of MacMimic. No MacMimic code, data or weights are included.

The included neural weights were trained locally from scratch on simulated frames. The included video and image are simulator captures, not physical-robot evidence.
