// Sidebar Toggle
document.getElementById('toggleBtn').addEventListener('click', function() {
    document.getElementById('sidebar').classList.toggle('closed');
});

// Function to show/hide sections
function showSection(sectionId) {
    // Hide all sections
    const sections = document.querySelectorAll('.section-content');
    sections.forEach(section => {
        section.classList.remove('active');
        section.style.display = 'none';
    });

    // Show the selected section with a transition
    const activeSection = document.getElementById(sectionId);
    if (activeSection) {
        activeSection.style.display = 'block';
        setTimeout(() => activeSection.classList.add('active'), 10); // Delay to add the class after setting display block
    }
}

// =============================Scroll Detection for Pop-up Animation=============================

// Function to check if an element is in the viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Debounce function to limit the frequency of scroll event firing
function debounce(func, wait = 10, immediate = true) {
    let timeout;
    return function() {
        const context = this, args = arguments;
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
}

// Scroll event to trigger animations when student elements come into view
document.addEventListener('scroll', debounce(function() {
    const studentElements = document.querySelectorAll('.student .image-bg'); // Only select student backgrounds

    studentElements.forEach(function(element) {
        if (isInViewport(element)) {
            element.classList.add('active'); // Add 'active' class when the student image is in view
        }
    });
}));

// =============================Image Switching for Students=============================
// Switch between images in each student's container every 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    const students = document.querySelectorAll('.student .image-bg');
    
    students.forEach(student => {
        const images = student.querySelectorAll('img');
        let currentIndex = 0;

        // Immediately display the first image when the page loads
        images[currentIndex].classList.add('active');

        // Switch between images every 5 seconds
        setInterval(() => {
            // Remove 'active' class from all images
            images.forEach(img => img.classList.remove('active'));
            
            // Update index to the next image
            currentIndex = (currentIndex + 1) % images.length;
            
            // Add 'active' class to the next image
            images[currentIndex].classList.add('active');
        }, 500000000000);  // Switch every 5 seconds
    });
});

// =============================Highlight Active Sidebar Section=============================
// Function to check if an element is in the viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Function to highlight the active sidebar link
function highlightActiveSection() {
    const sections = document.querySelectorAll('.section-content');
    const links = document.querySelectorAll('.sidebar li');

    sections.forEach(section => {
        const sectionId = section.getAttribute('id');
        const link = document.querySelector(`a[href="#${sectionId}"]`).parentElement;

        if (isInViewport(section)) {
            links.forEach(link => link.classList.remove('active')); // Remove active class from all links
            link.classList.add('active'); // Add active class to the current link
        }
    });
}

// Add event listener for scrolling
window.addEventListener('scroll', highlightActiveSection);

// Function to show/hide sections and update the active section in the sidebar
function showSection(sectionId) {
    // Hide all sections
    const sections = document.querySelectorAll('.section-content');
    sections.forEach(section => {
        section.style.display = 'none';
    });

    // Show the selected section
    const activeSection = document.getElementById(sectionId);
    if (activeSection) {
        activeSection.style.display = 'block';
    }

    // Update active class in the sidebar
    const links = document.querySelectorAll('.sidebar li');
    links.forEach(link => link.classList.remove('active')); // Remove active class from all links

    const activeLink = document.querySelector(`a[href="#${sectionId}"]`).parentElement;
    if (activeLink) {
        activeLink.classList.add('active'); // Add active class to the clicked link
    }
}
