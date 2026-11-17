// Site header: the Log out button (present on every logged-in page).
// The CSRF token comes from the button's data-csrf-token attribute.
(function () {
  const button = document.getElementById("logout-button");
  if (!button) return;
  button.addEventListener("click", async () => {
    await fetch("/logout", {
      method: "POST",
      headers: { "X-CSRF-Token": button.dataset.csrfToken },
      credentials: "same-origin",
    });
    window.location.href = "/login";
  });
})();
