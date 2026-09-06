document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('image');
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files.length > 0) {
                const fileName = fileInput.files[0].name;
                console.log('Selected file:', fileName);
            }
        });
    }
});
