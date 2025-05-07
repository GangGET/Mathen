// DOM Elements
const dropZone = document.getElementById('dropZone');
const uploadModal = document.getElementById('uploadModal');
const uploadTrigger = document.getElementById('uploadTrigger');
const chatInput = document.getElementById('chatInput');
const sendChatBtn = document.getElementById('sendButton');
const stopButton = document.getElementById("stopButton");//
const chatMessages = document.getElementById('chatMessages');
const filePreview = document.getElementById('filePreview');
const progressBar = document.getElementById('progressBar');
const progressContainer = document.getElementById('progressContainer');
const toast = document.getElementById('toast');
const historyList = document.getElementById('historyList');
const toggleSidebar = document.getElementById('toggleSidebar');
const showSidebarBtn = document.getElementById('showSidebar'); 
const mobileSidebarToggle = document.getElementById('mobileSidebarToggle');
const sidebar = document.querySelector('.history-sidebar');
const newChatBtn = document.querySelector('.new-chat-btn');
const pdfUpload = document.getElementById('pdfUpload');
const uploadBtn = document.getElementById('uploadBtn');
const pdfViewer = document.getElementById('pdfViewer');
const pdfModal = document.getElementById('pdfModal');
const pdfList = document.getElementById('pdfList');
const viewPdfBtn = document.getElementById('viewPdfBtn');

// Debug element references
console.log('Element References:');
console.log('dropZone:', dropZone);
console.log('uploadModal:', uploadModal);
console.log('uploadTrigger:', uploadTrigger);
console.log('filePreview:', filePreview);
console.log('toast:', toast);

// Bootstrap instances
let toastInstance;
let modalInstance;
let controller = new AbortController();//
// Initialize Bootstrap components immediately
if (toast) {
    toastInstance = new bootstrap.Toast(toast);
    console.log('Toast instance created');
} else {
    console.error('Toast element not found');
}

if (uploadModal) {
    modalInstance = new bootstrap.Modal(uploadModal);
    console.log('Modal instance created');
} else {
    console.error('Upload modal element not found');
}

// Connect the file input to the handleFiles function
const fileInput = document.getElementById('fileInput');
if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        console.log('File input change event:', e.target.files);
        handleFiles(e.target.files);
    });
    console.log('File input event listener added');
}

// Document ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOMContentLoaded event triggered');
    setupEventListeners();
});

// Setup all event listeners
function setupEventListeners() {
    console.log('Setting up event listeners');
    
    if (uploadTrigger) {
        uploadTrigger.addEventListener('click', () => {
            console.log('Upload trigger clicked');
            if (modalInstance) {
                modalInstance.show();
                console.log('Modal shown');
            } else {
                console.error('modalInstance is undefined');
            }
        });
        console.log('Upload trigger event listener added');
    } else {
        console.error('Upload trigger element not found');
    }
    
    // Setup command buttons
    setupCommandButtons();
    
    setupDragAndDrop();
    setupChatListeners();
    setupDeleteButtons();
    
    // PDF Upload and View Functionality
    uploadBtn.addEventListener('click', function() {
        if (pdfUpload.files.length > 0) {
            const file = pdfUpload.files[0];
            const formData = new FormData();
            formData.append('pdf', file);
            
            // Show loading message
            appendMessage('system', 'Uploading and processing your PDF...');
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Close upload modal
                    const uploadModalInstance = bootstrap.Modal.getInstance(uploadModal);
                    uploadModalInstance.hide();
                    
                    // Create object URL for viewing
                    const objectURL = URL.createObjectURL(file);
                    localStorage.setItem('uploadedPDF', objectURL);
                    
                    // Open PDF viewer modal
                    pdfViewer.src = objectURL;
                    const pdfModalInstance = new bootstrap.Modal(pdfModal);
                    pdfModalInstance.show();
                    
                    // Show success message with instructions
                    appendMessage('system', 'PDF uploaded and processed successfully! You can now use:\n- /summary to get a summary of the document\n- /search to search within the document');
                } else {
                    appendMessage('system', 'Error: ' + (data.error || 'Failed to upload PDF'));
                }
            })
            .catch(error => {
                console.error('Error:', error);
                appendMessage('system', 'Error uploading PDF. Please try again.');
            });
        } else {
            appendMessage('system', 'Please select a PDF file first.');
        }
    });

    // Clean up object URL when PDF modal is closed
    pdfModal.addEventListener('hidden.bs.modal', function () {
        const objectURL = localStorage.getItem('uploadedPDF');
        if (objectURL) {
            URL.revokeObjectURL(objectURL);
            localStorage.removeItem('uploadedPDF');
        }
        pdfViewer.src = '';
    });

    // PDF List Functionality
    viewPdfBtn.addEventListener('click', function() {
        const pdfList = document.getElementById('pdfList');
        
        // Toggle PDF list visibility
        if (pdfList.style.display === 'none') {
            // Fetch and display PDF list
            fetch('/list_pdfs')
                .then(response => response.json())
                .then(files => {
                    pdfList.innerHTML = ''; // Clear existing list
                    
                    if (files.length === 0) {
                        pdfList.innerHTML = '<p class="text-muted">No PDFs uploaded yet</p>';
                    } else {
                        files.forEach(file => {
                            const link = document.createElement('div');
                            link.className = 'pdf-list-item';
                            link.style.padding = '5px';
                            link.style.cursor = 'pointer';
                            link.style.borderBottom = '1px solid #eee';
                            
                            const icon = document.createElement('i');
                            icon.className = 'fas fa-file-pdf me-2';
                            icon.style.color = '#dc3545';
                            
                            link.appendChild(icon);
                            link.appendChild(document.createTextNode(file));
                            
                            link.addEventListener('mouseover', function() {
                                this.style.backgroundColor = '#f8f9fa';
                            });
                            
                            link.addEventListener('mouseout', function() {
                                this.style.backgroundColor = 'transparent';
                            });
                            
                            link.addEventListener('click', function() {
                                window.open(`/view_pdf?filename=${encodeURIComponent(file)}`, '_blank');
                            });
                            
                            pdfList.appendChild(link);
                        });
                    }
                    pdfList.style.display = 'block';
                })
                .catch(error => {
                    console.error('Error fetching PDF list:', error);
                    showToast('Error fetching PDF list', 'error');
                });
        } else {
            pdfList.style.display = 'none';
        }
    });

    // Close PDF list when clicking outside
    document.addEventListener('click', function(event) {
        const pdfList = document.getElementById('pdfList');
        const viewPdfBtn = document.getElementById('viewPdfBtn');
        
        if (!pdfList.contains(event.target) && !viewPdfBtn.contains(event.target)) {
            pdfList.style.display = 'none';
        }
    });
}

// Setup command buttons
function setupCommandButtons() {
    console.log('Setting up command buttons');
    const commandButtons = document.querySelectorAll('.command-btn');
    commandButtons.forEach(button => {
        const command = button.getAttribute('data-command');
        button.addEventListener('click', () => {
            console.log('Command button clicked:', command);
            handleCommand(command);
        });
        console.log('Added event listener for command button:', command);
    });
}

// Setup chat-related event listeners
function setupChatListeners() {
    if (sendChatBtn) {
        sendChatBtn.addEventListener('click', handleChat);
        console.log('Send chat button event listener added');
    }
    
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleChat();
            }
        });
        console.log('Chat input keydown event listener added');
    }
    if (stopButton) {
        stopButton.addEventListener("click", stopProcessing);
        console.log("Stop button event listener added");
    }
}


// Application state
let pdfUploaded = false;
let uploadInProgress = false;
let currentCommand = null;

// Available commands
const COMMANDS = {
    summary: {
        command: '/summary',
        description: 'Generate a summary of the PDF'
    },
    search: {
        command: '/search',
        description: 'Search for specific content in the PDF'
    },
    download: {
        command: '/download',
        description: 'Download the current summary as PDF'
    },
    help: {
        command: '/help',
        description: 'Show available commands'
    }
};

// Chat History Management
let chatHistory = JSON.parse(localStorage.getItem('chatHistory') || '[]');
let currentChatId = null;

function createNewChat() {
    const chatId = Date.now().toString();
    const chat = {
        id: chatId,
        title: 'New Chat',
        messages: []
    };
    chatHistory.unshift(chat);
    saveChatHistory();
    currentChatId = chatId;
    updateHistoryList();
    clearChatMessages();
    return chat;
}

function updateChatTitle(chatId, firstMessage) {
    const chat = chatHistory.find(c => c.id === chatId);
    if (chat) {
        // Create a title from the first message (max 30 chars)
        chat.title = firstMessage.length > 30 ? firstMessage.substring(0, 30) + '...' : firstMessage;
        saveChatHistory();
        updateHistoryList();
    }
}

async function deleteChat(chatId) {
    try {
        chatHistory = chatHistory.filter(c => c.id !== chatId);
        saveChatHistory();
        if (currentChatId === chatId) {
            if (chatHistory.length > 0) {
                currentChatId = chatHistory[0].id;
                loadChat(currentChatId);
            } else {
                createNewChat();
            }
        }
        updateHistoryList();
        showToast('Chat deleted successfully', 'success');
    } catch (error) {
        showToast('Error deleting chat: ' + error.message, 'error');
    }
}

function saveChatHistory() {
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
}

function loadChat(chatId) {
    const chat = chatHistory.find(c => c.id === chatId);
    if (chat) {
        currentChatId = chatId;
        clearChatMessages();
        chat.messages.forEach(msg => {
            appendMessage(msg.content, msg.isUser);
        });
        updateHistoryList();
    }
}

function updateHistoryList() {
    historyList.innerHTML = '';
    
    chatHistory.forEach(chat => {
        const item = document.createElement('div');
        item.className = `history-item${chat.id === currentChatId ? ' active' : ''}`;
        item.innerHTML = `
            <i class="fas fa-comments"></i>
            <div class="title-container">
                <span class="title" contenteditable="true">${chat.title}</span>
                <button class="edit-btn" data-chat-id="${chat.id}" title="Edit chat name">
                    <i class="fas fa-edit"></i>
                </button>
            </div>
            <button class="delete-btn" data-chat-id="${chat.id}" title="Delete chat">
                <i class="fas fa-trash"></i>
            </button>
        `;
        
        const titleElement = item.querySelector('.title');
        titleElement.addEventListener('blur', () => {
            const newTitle = titleElement.textContent.trim();
            if (newTitle) {
                const chatToUpdate = chatHistory.find(c => c.id === chat.id);
                if (chatToUpdate) {
                    chatToUpdate.title = newTitle;
                    saveChatHistory();
                }
            } else {
                titleElement.textContent = chat.title; // Revert if empty
            }
        });

        titleElement.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                titleElement.blur();
            }
        });
        
        const deleteBtn = item.querySelector('.delete-btn');
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (confirm('Are you sure you want to delete this chat?')) {
                deleteChat(chat.id);
            }
        });
        
        const editBtn = item.querySelector('.edit-btn');
        editBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            titleElement.focus();
        });
        
        item.addEventListener('click', (e) => {
            if (!e.target.closest('.edit-btn') && !e.target.closest('.delete-btn')) {
                loadChat(chat.id);
            }
        });
        
        historyList.appendChild(item);
    });
}

function saveMessage(content, isUser) {
    if (!currentChatId) {
        const chat = createNewChat();
        currentChatId = chat.id;
    }
    
    const chat = chatHistory.find(c => c.id === currentChatId);
    if (chat) {
        chat.messages.push({ content, isUser });
        if (chat.messages.length === 1 && isUser) {
            updateChatTitle(currentChatId, content);
        }
        saveChatHistory();
    }
}

// Show toast message
function showToast(message, type = 'info') {
    console.log('Showing toast:', message, type);
    
    // Check if toast element exists
    if (!toast) {
        console.error('Toast element not found');
        alert(message); // Fallback to alert if toast element doesn't exist
        return;
    }
    
    // Check if Bootstrap toast instance exists
    if (!toastInstance) {
        console.error('Toast instance not found, creating new one');
        toastInstance = new bootstrap.Toast(toast);
    }
    
    // Update toast content and style
    toast.classList.remove('bg-success', 'bg-danger', 'bg-info');
    toast.classList.add(`bg-${type}`, 'text-white');
    
    const toastBody = toast.querySelector('.toast-body');
    if (toastBody) {
        toastBody.textContent = message;
    } else {
        console.error('Toast body element not found');
        toast.textContent = message; // Fallback
    }
    
    // Show the toast
    try {
        toastInstance.show();
        console.log('Toast shown');
    } catch (error) {
        console.error('Error showing toast:', error);
        alert(message); // Fallback
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Update file preview
function updateFilePreview(file) {
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    
    if (fileName && fileSize) {
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
    }
    
    filePreview.classList.add('show');
    progressContainer.classList.add('show');
}

// Update progress bar
function updateProgress(percent) {
    progressBar.style.width = `${percent}%`;
    progressBar.setAttribute('aria-valuenow', percent);
}

// Handle file upload
async function handleFiles(files) {
    if (!files || files.length === 0) {
        showToast('Please select at least one PDF file', 'error');
        return;
    }

    if (uploadInProgress) {
        showToast('Upload in progress, please wait', 'info');
        return;
    }

    const formData = new FormData();
    let validFiles = [];
    let invalidFiles = [];

    // Validate files
    Array.from(files).forEach(file => {
        if (file.type === 'application/pdf') {
            validFiles.push(file);
            formData.append('files[]', file);
        } else {
            invalidFiles.push(file.name);
        }
    });

    if (invalidFiles.length > 0) {
        showToast(`Invalid files: ${invalidFiles.join(', ')}. Only PDF files are allowed.`, 'error');
        if (validFiles.length === 0) return;
    }

    // Clear existing file list
    const fileList = document.getElementById('fileList');
    fileList.innerHTML = '';

    // Create file items with progress bars
    validFiles.forEach(file => {
        const fileItem = createFileItem(file);
        fileList.appendChild(fileItem);
    });

    // Show file list and set upload state
    fileList.style.display = 'block';
    uploadInProgress = true;

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }

        // Update UI to show success
        const statusBadges = fileList.querySelectorAll('.upload-status');
        const progressBars = fileList.querySelectorAll('.progress-bar');
        
        statusBadges.forEach(badge => {
            badge.innerHTML = '<span class="badge bg-success">Uploaded</span>';
        });

        progressBars.forEach(bar => {
            bar.style.width = '100%';
            bar.classList.add('bg-success');
        });

        showToast(data.message, 'success');
        pdfUploaded = true;
        enableInterface();

        // Close modal after successful upload
        if (modalInstance) {
            setTimeout(() => modalInstance.hide(), 1500);
        }

        // Add welcome message in chat
        appendMessage('Files uploaded successfully! You can now ask questions about the documents or use commands like /summary to generate a summary.', false);

    } catch (error) {
        showToast(error.message, 'error');
        
        // Update UI to show error
        const statusBadges = fileList.querySelectorAll('.upload-status');
        statusBadges.forEach(badge => {
            badge.innerHTML = '<span class="badge bg-danger">Failed</span>';
        });
    } finally {
        uploadInProgress = false;
    }
}

function createFileItem(file) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item mb-3';
    fileItem.innerHTML = `
        <div class="file-info d-flex align-items-center mb-2">
            <i class="fas fa-file-pdf text-primary me-2"></i>
            <div class="flex-grow-1">
                <h6 class="mb-0">${file.name}</h6>
                <small class="text-muted">${formatFileSize(file.size)}</small>
            </div>
            <div class="upload-status">
                <span class="badge bg-info">Uploading...</span>
            </div>
            <button class="btn btn-sm btn-danger delete-btn" data-filename="${file.name}">
                <i class="fas fa-trash"></i>
            </button>
        </div>
        <div class="progress" style="height: 4px;">
            <div class="progress-bar" role="progressbar" style="width: 0%"></div>
        </div>
    `;
    return fileItem;
}

function setupDeleteButtons() {
    document.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const filename = e.currentTarget.dataset.filename;
            if (confirm(`Are you sure you want to delete ${filename}?`)) {
                try {
                    const response = await fetch('/delete_file', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ filename })
                    });

                    const data = await response.json();
                    if (!response.ok) {
                        throw new Error(data.error || 'Delete failed');
                    }

                    // Remove the file item from the list
                    const fileItem = e.currentTarget.closest('.file-item');
                    fileItem.remove();
                    showToast(data.message, 'success');
                } catch (error) {
                    showToast(error.message, 'error');
                }
            }
        });
    });
}

// Enable interface after successful upload
function enableInterface() {
    // Update application state
    pdfUploaded = true;
    
    // Enable chat interface
    chatInput.disabled = false;
    
    // Find and enable the send button (fixing potential ID mismatch)
    const sendButton = document.getElementById('sendButton');
    if (sendButton) {
        sendButton.disabled = false;
    }
    if (sendChatBtn) {
        sendChatBtn.disabled = false;
    }
    
    // Enable command buttons
    const commandButtons = document.querySelectorAll('.command-btn');
    commandButtons.forEach(button => {
        button.disabled = false;
    });
    
    // Show success indication in the UI
    if (progressContainer) {
        progressContainer.classList.add('show');
    }
}

// Handle commands
async function handleCommand(command, args = ' ') {
    if (!pdfUploaded && command !== '/help') {
        appendMessage('Please upload a PDF first.', false);
        saveMessage('Please upload a PDF first.', false);
        return;
    }
    controller = new AbortController(); // Reset controller before each command

    switch (command) {
        case '/summary':
            try {
                appendMessage('Generating summary...', false);
                saveMessage('Generating summary...', false);
                const response = await fetch('/generate_summary', {
                    method: 'POST',
                    signal: controller.signal // Allow stopping request
                });
                
                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.error || 'Failed to generate summary');
                }
                
                const data = await response.json();
                appendMessage(data.summary, false);
                saveMessage(data.summary, false);
            } catch (error) {
                appendMessage(`Error: ${error.message}`, false);
                saveMessage(`Error: ${error.message}`, false);
            }
            break;

        case '/search':
            if (!args) {
                appendMessage('Please provide a search term. Example: `/search artificial intelligence`', false);
                saveMessage('Please provide a search term. Example: `/search artificial intelligence`', false);
                return;
            }
            try {
                appendMessage(`Searching for: "${args}"...`, false);
                saveMessage(`Searching for: "${args}"...`, false);
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ query: args }),
                    signal: controller.signal // Allow stopping request
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || 'Search failed');
                }

                const results = data.results;
                if (results && results.length > 0) {
                    let message = '**Search Results:**\n\n';
                    results.forEach((result, index) => {
                        message += `${index + 1}. ${result}\n\n`;
                    });
                    appendMessage(message, false);
                    saveMessage(message, false);
                } else {
                    appendMessage('No results found.', false);
                    saveMessage('No results found.', false);
                }
            } catch (error) {
                appendMessage(`Error: ${error.message}`, false);
                saveMessage(`Error: ${error.message}`, false);
            }
            break;

        case '/download':
            try {
                appendMessage('Downloading summary...', false);
                saveMessage('Downloading summary...', false);
                window.location.href = '/download_summary';
                appendMessage('Download started!', false);
                saveMessage('Download started!', false);
            } catch (error) {
                appendMessage(`Error: ${error.message}`, false);
                saveMessage(`Error: ${error.message}`, false);
            }
            break;

        case '/help':
            showHelp();
            break;

        default:
            appendMessage('Unknown command. Type `/help` to see available commands.', false);
            saveMessage('Unknown command. Type `/help` to see available commands.', false);
    }
}

// Show help message
function showHelp() {
    const helpMessage = `
**Available Commands:**

- \`/summary\`: Generate a summary of the uploaded PDF
- \`/search [term]\`: Search for specific content in the PDF
- \`/download\`: Download the current summary as PDF
- \`/help\`: Show this help message

**How to use:**
1. Upload a PDF using the "Upload PDF" button
2. Use commands or ask questions naturally
3. Use \`/summary\` to get an overview of the document
4. Use \`/search\` followed by keywords to find specific information
5. Use \`/download\` to save the generated summary as PDF
`;
    appendMessage(helpMessage, false);
    saveMessage(helpMessage, false);
}

// Handle chat input
async function handleChat() {
    const message = chatInput.value.trim();
    if (!message) return;

    if (!pdfUploaded && !message.startsWith('/help')) {
        appendMessage('Please upload a PDF first by clicking the "Upload PDF" button.', false);
        saveMessage('Please upload a PDF first by clicking the "Upload PDF" button.', false);
        return;
    }

    appendMessage(message, true);
    saveMessage(message, true);
    chatInput.value = '';

    if (message.startsWith('/')) {
        const [command, ...args] = message.split(' ');
        await handleCommand(command, args.join(' '));
    } else {
        try {
            controller = new AbortController(); // Reset the abort controller
            appendMessage('Processing...', false);
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message }),
                signal: controller.signal // Attach abort controller
            });

        
            if (!response.ok) {
                throw new Error(data.error || 'Failed to get response from server');
            }
            const data = await response.json();

            // Remove the "Processing..." message
            chatMessages.removeChild(chatMessages.lastChild);
            
            appendMessage(data.response, false);
            saveMessage(data.response, false);
        } catch (error) {
            // Remove the "Processing..." message if it exists
            if (chatMessages.lastChild && chatMessages.lastChild.textContent.includes('Processing...')) {
                chatMessages.removeChild(chatMessages.lastChild);
            }

            let errorMessage = 'Error: ';
            if (error.name === 'TypeError' && error.message === 'Failed to fetch') {
                errorMessage += 'Unable to connect to server. Please check your internet connection and try again.';
            } else {
                errorMessage += error.message;
            }
            appendMessage(errorMessage, false);
            saveMessage(errorMessage, false);
            showToast(errorMessage, 'error');
        }
    }
}
function stopProcessing() {
    controller.abort(); // Abort the current request
    // Send a request to the backend to cancel processing
    fetch('/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    }).then(response => response.json())
      .then(data => console.log(data.message || data.error))
      .catch(error => console.error('Cancel request failed:', error));
}

// Handle fetch errors (Stops properly)
function handleFetchError(error) {
    if (chatMessages.lastChild && chatMessages.lastChild.textContent.includes('Processing...')) {
        chatMessages.removeChild(chatMessages.lastChild);
    }

    let errorMessage;
    if (error.name === 'AbortError') {
        errorMessage = 'Stopped.';
    } else if (error.name === 'TypeError' && error.message === 'Failed to fetch') {
        errorMessage = 'Error: Unable to connect to server. Please check your internet connection and try again.';
    } else {
        errorMessage = `Error: ${error.message}`;
    }

    appendMessage(errorMessage, false);
    saveMessage(errorMessage, false);
    showToast(errorMessage, 'error');
}

// Append message to chat
function appendMessage(message, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${isUser ? 'user-message' : ''}`;
    
    const iconDiv = document.createElement('div');
    iconDiv.className = 'message-icon';
    iconDiv.innerHTML = isUser ? 
        '<i class="fas fa-user"></i>' : 
        '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = marked.parse(message);
    
    messageDiv.appendChild(iconDiv);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Clear chat messages
function clearChatMessages() {
    chatMessages.innerHTML = '';
}

// Setup drag and drop event listeners
function setupDragAndDrop() {
    if (!dropZone) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const droppedFiles = e.dataTransfer.files;
        handleFiles(droppedFiles);
    }, false);

    dropZone.addEventListener('click', () => {
        fileInput.click();
    }, false);
}

// Prevent default drag behaviors
function preventDefaults(e) {
    e.preventDefault();
    e.stopImmediatePropagation();
}

toggleSidebar.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    showSidebarBtn.style.display = 'block'; 
});
showSidebarBtn.addEventListener('click', () => {
    sidebar.classList.remove('collapsed');
    showSidebarBtn.style.display = 'none'; // Hide new toggle button when sidebar is shown
});

mobileSidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('show');
});

newChatBtn.addEventListener('click', createNewChat);

// Initialize
if (chatHistory.length === 0) {
    createNewChat();
} else {
    currentChatId = chatHistory[0].id;
    loadChat(currentChatId);
}

// document.getElementById("viewPdfBtn").addEventListener("click", function () {
//     fetch("/list_pdfs")
//         .then(response => response.json())
//         .then(files => {
//             const pdfList = document.getElementById("pdfList");
//             pdfList.innerHTML = "";
//             pdfList.style.display = "block";
//             files.forEach(file => {
//                 let button = document.createElement("button");
//                 button.textContent = file;
//                 button.className = "pdf-button"; // Apply button style
//                 button.onclick = function() {
//                     window.open(`/view_pdf?filename=${file}`, "_blank");
//                 };
//                 pdfList.appendChild(button);
//                 pdfList.appendChild(document.createElement("br"));
//             });
//         });
// });

document.getElementById("chatInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        let userInput = this.value.trim().toLowerCase();
        if (userInput === "summary") {
            document.querySelector(".command-btn[data-command='/summary']").click();
            this.value = "";
            event.preventDefault();
        }
    }
});



document.getElementById("viewPdfBtn").addEventListener("click", function () {
    const pdfList = document.getElementById("pdfList");
    if (pdfList.style.display === "block") {
        pdfList.style.display = "none";
        return;
    }
    
    fetch("/list_pdfs")
        .then(response => response.json())
        .then(files => {
            pdfList.innerHTML = "";
            pdfList.style.display = "block";
            files.forEach(file => {
                let button = document.createElement("button");
                button.textContent = file;
                button.className = "pdf-button"; // Apply button style
                button.onclick = function() {
                    window.open(`/view_pdf?filename=${file}`, "_blank");
                };
                pdfList.appendChild(button);
                pdfList.appendChild(document.createElement("br"));
            });
        });
});
