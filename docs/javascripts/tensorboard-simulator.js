(() => {
  const CHART = Object.freeze({ left: 78, top: 55, width: 552, height: 260, maxLoss: 2.5 });
  const RUNS = Object.freeze({
    "0.0001": {
      train: [2.25, 2.05, 1.88, 1.72, 1.58, 1.46, 1.35, 1.25, 1.16, 1.08],
      validation: [2.30, 2.14, 1.99, 1.85, 1.73, 1.63, 1.54, 1.46, 1.39, 1.33],
      answer: "a",
      explanation: "Both losses decrease gradually. A small learning rate can make progress slow, so training may need more epochs.",
      status: "Both curves decrease gradually, which can mean progress is slow."
    },
    "0.001": {
      train: [1.70, 1.05, 0.68, 0.46, 0.32, 0.23, 0.17, 0.13, 0.10, 0.08],
      validation: [1.76, 1.11, 0.73, 0.50, 0.36, 0.27, 0.21, 0.17, 0.14, 0.12],
      answer: "b",
      explanation: "Both losses fall steadily and stay close in this illustration, which is the most useful of these three patterns.",
      status: "Both curves fall steadily and stay close in this illustration."
    },
    "0.01": {
      train: [1.90, 0.65, 1.35, 0.40, 1.10, 0.25, 0.90, 0.20, 0.70, 0.18],
      validation: [2.00, 1.15, 1.75, 0.90, 1.65, 0.72, 1.55, 0.62, 1.48, 0.55],
      answer: "c",
      explanation: "The losses jump up and down. A large learning rate can overshoot useful values and make updates unstable.",
      status: "Both curves jump up and down, which can mean updates are too large."
    }
  });
  const SVG_NS = "http://www.w3.org/2000/svg";

  function makeLine(values) {
    return values.map((value, index) => {
      const x = CHART.left + (index / (values.length - 1)) * CHART.width;
      const y = CHART.top + CHART.height - (value / CHART.maxLoss) * CHART.height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(" ");
  }

  function makePoints(values, className) {
    return values.map((value, index) => {
      const point = document.createElementNS(SVG_NS, "circle");
      const x = CHART.left + (index / (values.length - 1)) * CHART.width;
      const y = CHART.top + CHART.height - (value / CHART.maxLoss) * CHART.height;
      point.setAttribute("cx", x.toFixed(1));
      point.setAttribute("cy", y.toFixed(1));
      point.setAttribute("r", "3.5");
      point.setAttribute("class", className);
      return point;
    });
  }

  function clearQuizFeedback(quiz) {
    quiz.querySelectorAll(".quiz-answer-correct").forEach((label) => {
      label.classList.remove("quiz-answer-correct");
    });
    quiz.querySelectorAll("input").forEach((input) => {
      input.checked = false;
    });
    quiz.querySelector(".quiz-result").replaceChildren();
  }

  function bindTensorboardSimulators() {
    document.querySelectorAll("[data-tb-simulator]:not([data-simulator-bound])").forEach((simulator) => {
      simulator.dataset.simulatorBound = "true";
      const rateSelect = simulator.querySelector("[data-learning-rate]");
      const chart = simulator.querySelector(".tb-simulator-chart");
      const trainLine = chart.querySelector("[data-train-line]");
      const validationLine = chart.querySelector("[data-validation-line]");
      const trainPoints = chart.querySelector("[data-train-points]");
      const validationPoints = chart.querySelector("[data-validation-points]");
      const chartDescription = chart.querySelector("[data-chart-description]");
      const status = simulator.querySelector("[data-simulator-status]");
      const quiz = simulator.querySelector("[data-simulator-quiz]");

      function updateRun() {
        const rate = rateSelect.value;
        const run = RUNS[rate];
        trainLine.setAttribute("d", makeLine(run.train));
        validationLine.setAttribute("d", makeLine(run.validation));
        trainPoints.replaceChildren(...makePoints(run.train, "tb-point-train"));
        validationPoints.replaceChildren(...makePoints(run.validation, "tb-point-validation"));
        chartDescription.textContent = `Illustrative train and validation loss for learning rate ${rate} across ten epochs. ${run.status}`;
        status.textContent = `Illustrative data — learning rate ${rate}: ${run.status}`;
        quiz.dataset.answer = run.answer;
        quiz.dataset.explanation = run.explanation;
        clearQuizFeedback(quiz);
      }

      rateSelect.addEventListener("change", updateRun);
      updateRun();
    });
  }

  if (typeof document$ === "undefined") bindTensorboardSimulators();
  else document$.subscribe(bindTensorboardSimulators);
})();
