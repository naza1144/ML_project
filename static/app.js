const form = document.getElementById("predict-form");
const resultSection = document.getElementById("result");
const errorEl = document.getElementById("error");
const topCoffeeEl = document.getElementById("top-coffee");
const barsEl = document.getElementById("bars");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.classList.add("hidden");

  const payload = {
    weekday: document.getElementById("weekday").value,
    month: document.getElementById("month").value,
    time_of_day: document.getElementById("time_of_day").value,
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "เกิดข้อผิดพลาด");
    }

    topCoffeeEl.textContent = data.prediction;
    barsEl.innerHTML = "";

    data.probabilities.forEach(({ coffee_name, probability, in_top_k }) => {
      const pct = (probability * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "bar-row" + (in_top_k ? " bar-row-top-k" : "");
      row.innerHTML = `
        <span>${in_top_k ? "⭐ " : ""}${coffee_name}</span>
        <span class="bar-track"><span class="bar-fill" style="width: ${pct}%"></span></span>
        <span class="bar-pct">${pct}%</span>
      `;
      barsEl.appendChild(row);
    });

    resultSection.classList.remove("hidden");
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
    resultSection.classList.add("hidden");
  }
});
