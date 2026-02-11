/* ================================================================
   ATS Resume Analyzer — Client-side JS
   Dark mode, toasts, HTMX events, animated counters
   ================================================================ */

// ----- Dark Mode -----
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  const newTheme = isDark ? 'light' : 'dark';
  html.setAttribute('data-theme', newTheme);
  localStorage.setItem('ats-theme', newTheme);
  updateThemeIcons();
}

function updateThemeIcons() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  // Desktop icons
  const sun = document.getElementById('theme-sun');
  const moon = document.getElementById('theme-moon');
  // Mobile icons
  const sunM = document.getElementById('theme-sun-mobile');
  const moonM = document.getElementById('theme-moon-mobile');

  if (sun && moon) {
    sun.classList.toggle('hidden', !isDark);
    moon.classList.toggle('hidden', isDark);
  }
  if (sunM && moonM) {
    sunM.classList.toggle('hidden', !isDark);
    moonM.classList.toggle('hidden', isDark);
  }
}

// Init on load
document.addEventListener('DOMContentLoaded', updateThemeIcons);

// ----- Toast Notifications -----
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const icons = {
    success: '<svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5"/></svg>',
    error:   '<svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"/></svg>',
    info:    '<svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z"/></svg>',
    warning: '<svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"/></svg>',
  };

  const colors = {
    success: 'background: #2a8703; color: white;',
    error:   'background: #ea1100; color: white;',
    info:    'background: #0053e2; color: white;',
    warning: 'background: #995213; color: white;',
  };

  const toast = document.createElement('div');
  toast.className = 'toast animate-slide-up';
  toast.style.cssText = colors[type] || colors.info;
  toast.innerHTML = `${icons[type] || icons.info} <span>${message}</span>`;
  toast.setAttribute('role', 'alert');
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(8px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ----- Animated Counter -----
function animateCounter(element, target, duration = 1200) {
  const start = 0;
  const startTime = performance.now();
  const isFloat = target % 1 !== 0;

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = start + (target - start) * eased;
    element.textContent = isFloat ? current.toFixed(1) : Math.round(current);
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// Auto-animate counters when they enter viewport
function initCounters() {
  const counters = document.querySelectorAll('[data-counter]');
  if (!counters.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = parseFloat(el.dataset.counter);
        animateCounter(el, target);
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.3 });

  counters.forEach(el => observer.observe(el));
}

document.addEventListener('DOMContentLoaded', initCounters);

// ----- HTMX Events -----
document.addEventListener('htmx:afterRequest', function (event) {
  if (event.detail.failed) {
    let msg = 'Something went wrong. Please try again.';
    try {
      const data = JSON.parse(event.detail.xhr.responseText);
      msg = data.detail || msg;
    } catch {}
    showToast(msg, 'error');
  }
});

document.addEventListener('htmx:responseError', function () {
  showToast('Server error. Check if the backend is running.', 'error');
});

// Scroll to results + reinit counters after HTMX swap
document.addEventListener('htmx:afterSwap', function (event) {
  if (event.detail.target && event.detail.target.id === 'results-area') {
    setTimeout(() => {
      event.detail.target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      // Re-init any counters in swapped content
      initCounters();
    }, 100);
  }
});

// ----- Score Ring (SVG circular progress) -----
function createScoreRing(element, score, size = 80, strokeWidth = 6) {
  element.classList.add('score-ring');
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  let color = '#ea1100';
  if (score >= 80) color = '#2a8703';
  else if (score >= 65) color = '#0053e2';
  else if (score >= 50) color = '#995213';

  element.innerHTML = `
    <svg width="${size}" height="${size}">
      <circle class="score-ring-track" cx="${size/2}" cy="${size/2}" r="${radius}"
        fill="none" stroke-width="${strokeWidth}"/>
      <circle class="score-ring-fill" cx="${size/2}" cy="${size/2}" r="${radius}"
        fill="none" stroke="${color}" stroke-width="${strokeWidth}"
        stroke-dasharray="${circumference}" stroke-dashoffset="${circumference}"
        stroke-linecap="round"/>
    </svg>
    <div class="score-ring-label" style="font-size: ${size * 0.25}px; color: ${color}">${score}</div>
  `;

  // Animate after a small delay
  requestAnimationFrame(() => {
    const fill = element.querySelector('.score-ring-fill');
    if (fill) fill.style.strokeDashoffset = offset;
  });
}

// ----- Keyboard shortcuts -----
document.addEventListener('keydown', function(e) {
  // Press 'd' to toggle dark mode (when not typing)
  if (e.key === 'd' && !e.ctrlKey && !e.metaKey && !e.altKey) {
    const active = document.activeElement;
    const isTyping = active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable);
    if (!isTyping) {
      toggleTheme();
    }
  }
});
