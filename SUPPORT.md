# Support

When asking for help, include:

- Windows version
- Python version if installed
- The latest file in `Logs`
- The model folder layout under `Models`
- The input media extension
- Whether CPU or CUDA was selected

Common fixes:

- Run `setup.bat` again to repair the environment.
- Use complete model folders, not standalone weight files.
- Add `modelbench.json` for generic ONNX models.
- Use CPU mode if CUDA provider setup fails.
