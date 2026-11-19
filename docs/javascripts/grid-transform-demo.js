(() => {
  document.querySelectorAll("[data-grid-transform]:not([data-transform-bound])").forEach((demo) => {
    demo.dataset.transformBound = "true";
    const inputGrid = demo.querySelector(".conv-input-grid");
    const outputGrid = demo.querySelector(".conv-output-grid");
    const rows = Number(inputGrid.dataset.rows);
    const columns = Number(inputGrid.dataset.cols);
    const windowRows = Number(inputGrid.dataset.windowRows);
    const windowColumns = Number(inputGrid.dataset.windowCols);
    const stride = Number(inputGrid.dataset.stride);
    const mode = demo.dataset.gridTransform;
    const values = [...inputGrid.querySelectorAll("[data-value]")].map((cell) => Number(cell.dataset.value));
    const outputRows = Math.floor((rows - windowRows) / stride) + 1;
    const outputColumns = Math.floor((columns - windowColumns) / stride) + 1;
    const positions = [];

    for (let row = 0; row <= rows - windowRows; row += stride) {
      for (let column = 0; column <= columns - windowColumns; column += stride) {
        positions.push([row, column]);
      }
    }

    inputGrid.style.setProperty("--grid-cols", columns);
    outputGrid.style.setProperty("--grid-cols", outputColumns);
    const inputStage = demo.querySelector(".conv-input-stage");
    const outline = demo.querySelector(".conv-kernel-outline");
    const outputCells = [...outputGrid.querySelectorAll("[data-output-cell]")];
    const status = demo.querySelector(".conv-status");
    const runButton = demo.querySelector(".conv-run");
    const resetButton = demo.querySelector(".conv-reset");
    let runId = 0;

    const moveWindow = (step) => {
      const [row, column] = positions[step];
      const firstCell = inputGrid.children[row * columns + column].getBoundingClientRect();
      const lastCell = inputGrid.children[(row + windowRows - 1) * columns + column + windowColumns - 1].getBoundingClientRect();
      const stage = inputStage.getBoundingClientRect();
      outline.style.left = `${firstCell.left - stage.left}px`;
      outline.style.top = `${firstCell.top - stage.top}px`;
      outline.style.width = `${lastCell.right - firstCell.left}px`;
      outline.style.height = `${lastCell.bottom - firstCell.top}px`;
      outline.hidden = false;
    };

    const applyWindow = (step) => {
      const [row, column] = positions[step];
      const patch = [];
      for (let y = 0; y < windowRows; y += 1) {
        for (let x = 0; x < windowColumns; x += 1) {
          patch.push(values[(row + y) * columns + column + x]);
        }
      }
      const result = mode === "relu" ? Math.max(0, patch[0]) : Math.max(...patch);
      outputCells[step].textContent = result;
      outputCells[step].classList.add("is-filled");
      status.textContent = mode === "relu"
        ? `Step ${step + 1} of ${positions.length}: ReLU changes ${patch[0]} to ${result}.`
        : `Step ${step + 1} of ${positions.length}: max(${patch.join(", ")}) = ${result}.`;
    };

    runButton.addEventListener("click", async () => {
      const currentRun = ++runId;
      runButton.disabled = true;
      for (let step = 0; step < positions.length; step += 1) {
        if (currentRun !== runId) return;
        const [row, column] = positions[step];
        moveWindow(step);
        status.textContent = `Step ${step + 1} of ${positions.length}: move the highlighted window to output row ${Math.floor(row / stride) + 1}, column ${Math.floor(column / stride) + 1}.`;
        const delay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 120 : 650;
        await new Promise((resolve) => window.setTimeout(resolve, delay));
        if (currentRun !== runId) return;
        applyWindow(step);
        await new Promise((resolve) => window.setTimeout(resolve, 180));
      }
      if (currentRun === runId) {
        status.textContent = `Done: all ${positions.length} windows filled the ${outputRows} × ${outputColumns} output grid.`;
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
      status.textContent = `Ready. Run the ${positions.length} steps to fill the output.`;
      runButton.disabled = false;
    });
  });
})();
