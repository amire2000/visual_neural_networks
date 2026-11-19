function bindQuizzes() {
  document.querySelectorAll(".quiz:not([data-quiz-bound])").forEach((quiz) => {
    quiz.dataset.quizBound = "true";
    quiz.querySelector(".quiz-check").addEventListener("click", () => {
      const answer = quiz.querySelector("input:checked");
      const result = quiz.querySelector(".quiz-result");
      quiz.querySelectorAll(".quiz-answer-correct").forEach((label) => label.classList.remove("quiz-answer-correct"));
      result.replaceChildren();
      if (!answer) {
        result.textContent = "Choose an answer first.";
        return;
      }
      if (answer.value !== quiz.dataset.answer) {
        result.textContent = "Try again.";
        return;
      }
      answer.closest("label").classList.add("quiz-answer-correct");
      const status = document.createElement("strong");
      status.textContent = "✓ Correct.";
      const explanation = document.createElement("span");
      explanation.className = "quiz-explanation";
      explanation.textContent = quiz.dataset.explanation;
      result.append(status, explanation);
    });
  });
}

if (typeof document$ === "undefined") bindQuizzes();
else document$.subscribe(bindQuizzes);
