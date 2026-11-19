(() => {
  document.querySelectorAll("[data-convolution-demo]:not([data-demo-bound])").forEach((demo) => {
    demo.dataset.demoBound = "true";
    const inputGrid = demo.querySelector(".conv-input-grid");
    const kernelGrid = demo.querySelector(".conv-kernel-grid");
    const outputGrid = demo.querySelector(".conv-output-grid");
    const inputRows = Number(inputGrid.dataset.rows);
    const inputCols = Number(inputGrid.dataset.cols);
    const kernelRows = Number(kernelGrid.dataset.rows);
    const kernelCols = Number(kernelGrid.dataset.cols);
    const outputRows = inputRows - kernelRows + 1;
    const outputCols = inputCols - kernelCols + 1;
    const input = [...inputGrid.querySelectorAll("[data-value]")].map((cell) => Number(cell.dataset.value));
    const kernel = [...kernelGrid.querySelectorAll("[data-value]")].map((cell) => Number(cell.dataset.value));
    const outputCells = [...outputGrid.querySelectorAll("[data-output-cell]")];
    const positions = [];
    for (let row = 0; row < outputRows; row += 1) {
      for (let column = 0; column < outputCols; column += 1) positions.push([row, column]);
    }

    inputGrid.style.setProperty("--grid-cols", inputCols);
    kernelGrid.style.setProperty("--grid-cols", kernelCols);
    outputGrid.style.setProperty("--grid-cols", outputCols);
    const inputStage = demo.querySelector(".conv-input-stage");
    const outline = demo.querySelector(".conv-kernel-outline");
    const status = demo.querySelector(".conv-status");
    const runButton = demo.querySelector(".conv-run");
    const resetButton = demo.querySelector(".conv-reset");
    let runId = 0;

    const moveKernel = (step) => {
      const [row, column] = positions[step];
      const firstCell = inputGrid.children[row * inputCols + column].getBoundingClientRect();
      const lastCell = inputGrid.children[(row + kernelRows - 1) * inputCols + column + kernelCols - 1].getBoundingClientRect();
      const stage = inputStage.getBoundingClientRect();
      outline.style.left = `${firstCell.left - stage.left}px`;
      outline.style.top = `${firstCell.top - stage.top}px`;
      outline.style.width = `${lastCell.right - firstCell.left}px`;
      outline.style.height = `${lastCell.bottom - firstCell.top}px`;
      outline.hidden = false;
    };

    const fillOutput = (step) => {
      const [row, column] = positions[step];
      let response = 0;
      for (let y = 0; y < kernelRows; y += 1) {
        for (let x = 0; x < kernelCols; x += 1) {
          response += input[(row + y) * inputCols + column + x] * kernel[y * kernelCols + x];
        }
      }
      outputCells[step].textContent = response;
      outputCells[step].classList.add("is-filled");
      status.textContent = `Step ${step + 1} of ${positions.length}: output row ${row + 1}, column ${column + 1} is ${response}.`;
    };

    runButton.addEventListener("click", async () => {
      const currentRun = ++runId;
      runButton.disabled = true;
      for (let step = 0; step < positions.length; step += 1) {
        if (currentRun !== runId) return;
        const [row, column] = positions[step];
        moveKernel(step);
        status.textContent = `Step ${step + 1} of ${positions.length}: move the kernel to output row ${row + 1}, column ${column + 1}.`;
        const delay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 120 : 650;
        await new Promise((resolve) => window.setTimeout(resolve, delay));
        if (currentRun !== runId) return;
        fillOutput(step);
        await new Promise((resolve) => window.setTimeout(resolve, 180));
      }
      if (currentRun === runId) {
        status.textContent = `Done: all ${positions.length} filter responses fill the ${outputRows} × ${outputCols} output feature map.`;
        runButton.disabled = false;
      }
    });

    resetButton.addEventListener("click", () => {
      runId += 1;
      outline.hidden = true;
      outputCells.forEach((cell) => {
        cell.textContent = "—";
        cell.classList.remove("is-filled");
      });
      status.textContent = `Ready. Run the ${positions.length} kernel positions to fill the output.`;
      runButton.disabled = false;
    });
  });
})();
