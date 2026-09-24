/* =========================================================
   ROADGUARD AI
   Frontend JavaScript

   Features:
   - Image upload
   - Image preview
   - Drag & drop
   - Browser camera
   - Live AI detection
   - Frame sending to Flask
   ========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    () => {


    /* =====================================================
       IMAGE UPLOAD ELEMENTS
    ===================================================== */

    const uploadForm =
        document.getElementById(
            "uploadForm"
        );

    const uploadArea =
        document.getElementById(
            "uploadArea"
        );

    const imageInput =
        document.getElementById(
            "imageInput"
        );

    const uploadContent =
        document.getElementById(
            "uploadContent"
        );

    const previewContainer =
        document.getElementById(
            "previewContainer"
        );

    const imagePreview =
        document.getElementById(
            "imagePreview"
        );

    const previewFileName =
        document.getElementById(
            "previewFileName"
        );

    const removeImage =
        document.getElementById(
            "removeImage"
        );

    const fileInfo =
        document.getElementById(
            "fileInfo"
        );

    const fileName =
        document.getElementById(
            "fileName"
        );

    const fileSize =
        document.getElementById(
            "fileSize"
        );

    const analyzeButton =
        document.getElementById(
            "analyzeButton"
        );

    const buttonText =
        document.getElementById(
            "buttonText"
        );

    const buttonArrow =
        document.getElementById(
            "buttonArrow"
        );

    const processingPanel =
        document.getElementById(
            "processingPanel"
        );


    /* =====================================================
       IMAGE UPLOAD VARIABLES
    ===================================================== */

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp"
    ];

    const maxFileSize =
        25 * 1024 * 1024;

    let selectedFile = null;


    /* =====================================================
       CAMERA ELEMENTS
    ===================================================== */

    const cameraVideo =
        document.getElementById(
            "cameraVideo"
        );

    const cameraResult =
        document.getElementById(
            "cameraResult"
        );

    const cameraPlaceholder =
        document.getElementById(
            "cameraPlaceholder"
        );

    const cameraResultPlaceholder =
        document.getElementById(
            "cameraResultPlaceholder"
        );

    const cameraStatus =
        document.getElementById(
            "cameraStatus"
        );

    const cameraLiveBadge =
        document.getElementById(
            "cameraLiveBadge"
        );

    const cameraResolution =
        document.getElementById(
            "cameraResolution"
        );

    const startCameraButton =
        document.getElementById(
            "startCameraButton"
        );

    const startDetectionButton =
        document.getElementById(
            "startDetectionButton"
        );

    const stopCameraButton =
        document.getElementById(
            "stopCameraButton"
        );

    const aiProcessingLabel =
        document.getElementById(
            "aiProcessingLabel"
        );

    const livePotholeCount =
        document.getElementById(
            "livePotholeCount"
        );

    const liveConfidence =
        document.getElementById(
            "liveConfidence"
        );

    const liveProcessingStatus =
        document.getElementById(
            "liveProcessingStatus"
        );

    const cameraMessage =
        document.getElementById(
            "cameraMessage"
        );


    /* =====================================================
       CAMERA VARIABLES
    ===================================================== */

    let cameraStream = null;

    let cameraRunning = false;

    let detectionRunning = false;

    let frameProcessing = false;

    let cameraCanvas = null;

    let detectionTimeout = null;


    /*
     * CPU Faster R-CNN is relatively slow.
     *
     * We intentionally don't send frames
     * continuously at 30 FPS.
     *
     * Instead we process one frame,
     * wait for its response,
     * then capture another frame.
     */

    const CAMERA_FRAME_INTERVAL = 250;


    /* =====================================================
       IMAGE UPLOAD
    ===================================================== */

    if (uploadArea && imageInput) {

        uploadArea.addEventListener(
            "click",
            (event) => {

                if (
                    event.target.closest(
                        "#removeImage"
                    )
                ) {
                    return;
                }

                imageInput.click();

            }
        );

    }


    /* =====================================================
       FILE INPUT
    ===================================================== */

    if (imageInput) {

        imageInput.addEventListener(
            "change",
            (event) => {

                const files =
                    event.target.files;

                if (
                    !files ||
                    files.length === 0
                ) {
                    return;
                }

                handleFile(
                    files[0]
                );

            }
        );

    }


    /* =====================================================
       HANDLE IMAGE FILE
    ===================================================== */

    function handleFile(file) {

        if (
            !allowedTypes.includes(
                file.type
            )
        ) {

            showError(
                "Please select a valid image file."
            );

            resetFileInput();

            return;
        }


        if (
            file.size > maxFileSize
        ) {

            showError(
                "Image size must be less than 25 MB."
            );

            resetFileInput();

            return;
        }


        selectedFile = file;


        updateFileInformation(
            file
        );


        showPreview(
            file
        );


        if (analyzeButton) {

            analyzeButton.disabled =
                false;

        }

    }


    /* =====================================================
       UPDATE FILE INFO
    ===================================================== */

    function updateFileInformation(
        file
    ) {

        if (fileName) {

            fileName.textContent =
                file.name;

        }


        if (fileSize) {

            fileSize.textContent =
                formatFileSize(
                    file.size
                );

        }


        if (previewFileName) {

            previewFileName.textContent =
                file.name;

        }


        if (fileInfo) {

            fileInfo.classList.add(
                "active"
            );

        }

    }


    /* =====================================================
       IMAGE PREVIEW
    ===================================================== */

    function showPreview(
        file
    ) {

        if (
            !previewContainer ||
            !imagePreview
        ) {
            return;
        }


        const reader =
            new FileReader();


        reader.onload =
            (event) => {

                imagePreview.src =
                    event.target.result;

                previewContainer.classList.add(
                    "active"
                );


                if (uploadContent) {

                    uploadContent.style.display =
                        "none";

                }

            };


        reader.onerror =
            () => {

                showError(
                    "Unable to preview this image."
                );

            };


        reader.readAsDataURL(
            file
        );

    }


    /* =====================================================
       REMOVE IMAGE
    ===================================================== */

    if (removeImage) {

        removeImage.addEventListener(
            "click",
            (event) => {

                event.preventDefault();

                event.stopPropagation();

                resetUpload();

            }
        );

    }


    /* =====================================================
       RESET IMAGE UPLOAD
    ===================================================== */

    function resetUpload() {

        selectedFile = null;


        if (imageInput) {

            imageInput.value = "";

        }


        if (imagePreview) {

            imagePreview.src = "";

        }


        if (previewContainer) {

            previewContainer.classList.remove(
                "active"
            );

        }


        if (uploadContent) {

            uploadContent.style.display =
                "";

        }


        if (fileInfo) {

            fileInfo.classList.remove(
                "active"
            );

        }


        if (analyzeButton) {

            analyzeButton.disabled =
                true;

            analyzeButton.classList.remove(
                "loading"
            );

        }


        if (processingPanel) {

            processingPanel.classList.remove(
                "active"
            );

        }

    }


    /* =====================================================
       RESET INPUT
    ===================================================== */

    function resetFileInput() {

        selectedFile = null;

        if (imageInput) {

            imageInput.value = "";

        }

    }


    /* =====================================================
       DRAG & DROP
    ===================================================== */

    if (uploadArea) {

        uploadArea.addEventListener(
            "dragover",
            (event) => {

                event.preventDefault();

                uploadArea.classList.add(
                    "dragover"
                );

            }
        );


        uploadArea.addEventListener(
            "dragleave",
            (event) => {

                event.preventDefault();

                if (
                    !uploadArea.contains(
                        event.relatedTarget
                    )
                ) {

                    uploadArea.classList.remove(
                        "dragover"
                    );

                }

            }
        );


        uploadArea.addEventListener(
            "drop",
            (event) => {

                event.preventDefault();

                uploadArea.classList.remove(
                    "dragover"
                );


                const files =
                    event.dataTransfer.files;


                if (
                    !files ||
                    files.length === 0
                ) {
                    return;
                }


                const file =
                    files[0];


                handleFile(
                    file
                );


                try {

                    const dataTransfer =
                        new DataTransfer();

                    dataTransfer.items.add(
                        file
                    );

                    imageInput.files =
                        dataTransfer.files;

                } catch (error) {

                    console.warn(
                        "Could not synchronize file input:",
                        error
                    );

                }

            }
        );

    }


    /* =====================================================
       IMAGE FORM SUBMIT
    ===================================================== */

    if (uploadForm) {

        uploadForm.addEventListener(
            "submit",
            (event) => {

                if (!selectedFile) {

                    event.preventDefault();

                    showError(
                        "Please select a road image first."
                    );

                    return;

                }


                if (analyzeButton) {

                    analyzeButton.disabled =
                        true;

                    analyzeButton.classList.add(
                        "loading"
                    );

                }


                if (buttonText) {

                    buttonText.textContent =
                        "Analyzing...";

                }


                if (buttonArrow) {

                    buttonArrow.style.display =
                        "none";

                }


                if (processingPanel) {

                    processingPanel.classList.add(
                        "active"
                    );

                }


                startProcessingAnimation();

            }
        );

    }


    /* =====================================================
       IMAGE PROCESSING ANIMATION
    ===================================================== */

    let processingInterval = null;


    function startProcessingAnimation() {

        const steps =
            document.querySelectorAll(
                ".processing-step"
            );


        if (!steps.length) {
            return;
        }


        steps.forEach(
            (step) => {

                step.classList.remove(
                    "active"
                );

            }
        );


        let currentStep = 0;


        steps[
            currentStep
        ].classList.add(
            "active"
        );


        if (processingInterval) {

            clearInterval(
                processingInterval
            );

        }


        processingInterval =
            setInterval(
                () => {

                    if (
                        currentStep <
                        steps.length - 1
                    ) {

                        currentStep++;

                        steps[
                            currentStep
                        ].classList.add(
                            "active"
                        );

                    } else {

                        clearInterval(
                            processingInterval
                        );

                    }

                },
                900
            );

    }


    /* =====================================================
       CAMERA — START
    ===================================================== */

    if (startCameraButton) {

        startCameraButton.addEventListener(
            "click",
            startCamera
        );

    }


    async function startCamera() {

        try {

            showCameraMessage(
                "Requesting camera permission..."
            );


            /*
             * Ask browser for camera access.
             */

            cameraStream =
                await navigator.mediaDevices.getUserMedia({

                    video: {
                        facingMode: {
                            ideal: "environment"
                        },

                        width: {
                            ideal: 1280
                        },

                        height: {
                            ideal: 720
                        }
                    },

                    audio: false

                });


            /*
             * Connect camera to video element.
             */

            cameraVideo.srcObject =
                cameraStream;


            await cameraVideo.play();


            cameraRunning = true;


            /*
             * Create canvas once.
             */

            if (!cameraCanvas) {

                cameraCanvas =
                    document.createElement(
                        "canvas"
                    );

            }


            /*
             * UI update.
             */

            if (cameraPlaceholder) {

                cameraPlaceholder.classList.add(
                    "hidden"
                );

            }


            if (cameraLiveBadge) {

                cameraLiveBadge.classList.add(
                    "active"
                );

            }


            if (cameraStatus) {

                cameraStatus.classList.add(
                    "online"
                );

                cameraStatus.innerHTML =
                    `
                    <span></span>
                    Camera Active
                    `;

            }


            if (cameraResolution) {

                cameraResolution.textContent =
                    `${cameraVideo.videoWidth} × ${cameraVideo.videoHeight}`;

            }


            if (startCameraButton) {

                startCameraButton.disabled =
                    true;

            }


            if (startDetectionButton) {

                startDetectionButton.disabled =
                    false;

            }


            if (stopCameraButton) {

                stopCameraButton.disabled =
                    false;

            }


            showCameraMessage(
                "Camera connected successfully."
            );

        } catch (error) {

            console.error(
                "Camera error:",
                error
            );


            cameraRunning = false;


            if (
                error.name ===
                "NotAllowedError"
            ) {

                showCameraMessage(
                    "Camera permission was denied. Please allow camera access in your browser."
                );

            } else if (
                error.name ===
                "NotFoundError"
            ) {

                showCameraMessage(
                    "No camera was found on this device."
                );

            } else {

                showCameraMessage(
                    "Could not access the camera: "
                    + error.message
                );

            }

        }

    }


    /* =====================================================
       CAMERA — START AI DETECTION
    ===================================================== */

    if (startDetectionButton) {

        startDetectionButton.addEventListener(
            "click",
            () => {

                if (!cameraRunning) {

                    showCameraMessage(
                        "Start the camera first."
                    );

                    return;

                }


                if (detectionRunning) {

                    stopDetection();

                } else {

                    startDetection();

                }

            }
        );

    }


    /* =====================================================
       START DETECTION LOOP
    ===================================================== */

    function startDetection() {

        if (!cameraRunning) {
            return;
        }


        detectionRunning = true;


        if (startDetectionButton) {

            startDetectionButton.innerHTML =
                `
                <span>■</span>
                Stop Detection
                `;

            startDetectionButton.classList.add(
                "active"
            );

        }


        if (aiProcessingLabel) {

            aiProcessingLabel.textContent =
                "RUNNING";

            aiProcessingLabel.classList.add(
                "active"
            );

        }


        if (liveProcessingStatus) {

            liveProcessingStatus.textContent =
                "RUNNING";

        }


        showCameraMessage(
            "AI detection is running..."
        );


        processCameraFrame();

    }


    /* =====================================================
       STOP DETECTION
    ===================================================== */

    function stopDetection() {

        detectionRunning = false;

        frameProcessing = false;


        if (detectionTimeout) {

            clearTimeout(
                detectionTimeout
            );

            detectionTimeout = null;

        }


        if (startDetectionButton) {

            startDetectionButton.innerHTML =
                `
                <span>AI</span>
                Start Detection
                `;

            startDetectionButton.classList.remove(
                "active"
            );

            startDetectionButton.disabled =
                !cameraRunning;

        }


        if (aiProcessingLabel) {

            aiProcessingLabel.textContent =
                "STANDBY";

            aiProcessingLabel.classList.remove(
                "active"
            );

        }


        if (liveProcessingStatus) {

            liveProcessingStatus.textContent =
                "STANDBY";

        }

    }


    /* =====================================================
       PROCESS CAMERA FRAME
    ===================================================== */

    async function processCameraFrame() {

        if (
            !cameraRunning ||
            !detectionRunning
        ) {

            return;

        }


        /*
         * Don't send a second frame while
         * previous AI inference is running.
         */

        if (frameProcessing) {

            detectionTimeout =
                setTimeout(
                    processCameraFrame,
                    CAMERA_FRAME_INTERVAL
                );

            return;

        }


        frameProcessing = true;


        try {

            /*
             * Make sure video has dimensions.
             */

            if (
                cameraVideo.videoWidth === 0 ||
                cameraVideo.videoHeight === 0
            ) {

                frameProcessing = false;

                detectionTimeout =
                    setTimeout(
                        processCameraFrame,
                        500
                    );

                return;

            }


            /*
             * Capture current camera frame.
             */

            cameraCanvas.width =
                cameraVideo.videoWidth;

            cameraCanvas.height =
                cameraVideo.videoHeight;


            const context =
                cameraCanvas.getContext(
                    "2d"
                );


            context.drawImage(
                cameraVideo,
                0,
                0,
                cameraCanvas.width,
                cameraCanvas.height
            );


            /*
             * Convert frame to JPEG blob.
             */

            const blob =
                await canvasToBlob(
                    cameraCanvas,
                    "image/jpeg",
                    0.65
                );


            /*
             * Create multipart request.
             */

            const formData =
                new FormData();


            formData.append(
                "frame",
                blob,
                "camera_frame.jpg"
            );


            /*
             * Send frame to Flask.
             */

            const response =
                await fetch(
                    "/camera/detect",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            /*
             * Parse response.
             */

            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Camera detection failed."
                );

            }


            if (!data.success) {

                throw new Error(
                    data.error ||
                    "AI detection failed."
                );

            }


            /*
             * Display AI processed image.
             */

            if (cameraResult) {

                cameraResult.src =
                    data.image;

                cameraResult.classList.add(
                    "active"
                );

            }


            if (cameraResultPlaceholder) {

                cameraResultPlaceholder.classList.add(
                    "hidden"
                );

            }


            /*
             * Update statistics.
             */

            updateLiveStatistics(
                data
            );


        } catch (error) {

            console.error(
                "Frame detection error:",
                error
            );


            showCameraMessage(
                "Detection error: "
                + error.message
            );


        } finally {

            frameProcessing = false;


            /*
             * Capture next frame only
             * after current request finishes.
             */

            if (
                cameraRunning &&
                detectionRunning
            ) {

                detectionTimeout =
                    setTimeout(
                        processCameraFrame,
                        CAMERA_FRAME_INTERVAL
                    );

            }

        }

    }


    /* =====================================================
       CANVAS -> BLOB
    ===================================================== */

    function canvasToBlob(
        canvas,
        type,
        quality
    ) {

        return new Promise(
            (resolve, reject) => {

                canvas.toBlob(
                    (blob) => {

                        if (blob) {

                            resolve(
                                blob
                            );

                        } else {

                            reject(
                                new Error(
                                    "Could not create camera frame."
                                )
                            );

                        }

                    },
                    type,
                    quality
                );

            }
        );

    }


    /* =====================================================
       UPDATE LIVE STATISTICS
    ===================================================== */

    function updateLiveStatistics(
        data
    ) {

        if (livePotholeCount) {

            livePotholeCount.textContent =
                data.detection_count;

        }


        if (liveConfidence) {

            const confidence =
                data.average_confidence || 0;

            liveConfidence.textContent =
                `${(
                    confidence * 100
                ).toFixed(1)}%`;

        }


        if (liveProcessingStatus) {

            liveProcessingStatus.textContent =
                "ACTIVE";

        }


        if (aiProcessingLabel) {

            aiProcessingLabel.textContent =
                "AI ACTIVE";

        }

    }


    /* =====================================================
       CAMERA — STOP
    ===================================================== */

    if (stopCameraButton) {

        stopCameraButton.addEventListener(
            "click",
            stopCamera
        );

    }


    function stopCamera() {

        /*
         * Stop detection first.
         */

        stopDetection();


        /*
         * Stop all camera tracks.
         */

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(
                    (track) => {
                        track.stop();
                    }
                );

        }


        cameraStream = null;

        cameraRunning = false;


        /*
         * Remove video source.
         */

        if (cameraVideo) {

            cameraVideo.srcObject =
                null;

        }


        /*
         * UI
         */

        if (cameraPlaceholder) {

            cameraPlaceholder.classList.remove(
                "hidden"
            );

        }


        if (cameraLiveBadge) {

            cameraLiveBadge.classList.remove(
                "active"
            );

        }


        if (cameraStatus) {

            cameraStatus.classList.remove(
                "online"
            );

            cameraStatus.innerHTML =
                `
                <span></span>
                Camera Offline
                `;

        }


        if (startCameraButton) {

            startCameraButton.disabled =
                false;

        }


        if (startDetectionButton) {

            startDetectionButton.disabled =
                true;

        }


        if (stopCameraButton) {

            stopCameraButton.disabled =
                true;

        }


        if (cameraResolution) {

            cameraResolution.textContent =
                "--";

        }


        if (liveProcessingStatus) {

            liveProcessingStatus.textContent =
                "STANDBY";

        }


        if (aiProcessingLabel) {

            aiProcessingLabel.textContent =
                "STANDBY";

            aiProcessingLabel.classList.remove(
                "active"
            );

        }


        showCameraMessage(
            "Camera stopped."
        );

    }


    /* =====================================================
       CAMERA MESSAGE
    ===================================================== */

    function showCameraMessage(
        message
    ) {

        if (!cameraMessage) {
            return;
        }


        cameraMessage.textContent =
            message;


        cameraMessage.classList.add(
            "active"
        );


        clearTimeout(
            showCameraMessage.timeout
        );


        showCameraMessage.timeout =
            setTimeout(
                () => {

                    cameraMessage.classList.remove(
                        "active"
                    );

                },
                3500
            );

    }


    /* =====================================================
       FILE SIZE
    ===================================================== */

    function formatFileSize(
        bytes
    ) {

        if (bytes === 0) {

            return "0 Bytes";

        }


        const units = [
            "Bytes",
            "KB",
            "MB",
            "GB"
        ];


        const index =
            Math.floor(
                Math.log(bytes) /
                Math.log(1024)
            );


        const size =
            bytes /
            Math.pow(
                1024,
                index
            );


        return (
            size.toFixed(
                index === 0
                    ? 0
                    : 2
            )
            + " "
            + units[index]
        );

    }


    /* =====================================================
       ERROR
    ===================================================== */

    function showError(
        message
    ) {

        const oldError =
            document.querySelector(
                ".upload-error"
            );


        if (oldError) {

            oldError.remove();

        }


        const error =
            document.createElement(
                "div"
            );


        error.className =
            "upload-error";


        error.textContent =
            message;


        error.style.marginTop =
            "12px";

        error.style.padding =
            "12px 15px";

        error.style.border =
            "1px solid rgba(255,107,107,0.2)";

        error.style.borderRadius =
            "10px";

        error.style.background =
            "rgba(255,107,107,0.06)";

        error.style.color =
            "#ff8b8b";

        error.style.fontSize =
            "12px";

        error.style.textAlign =
            "center";


        if (uploadArea) {

            uploadArea.insertAdjacentElement(
                "afterend",
                error
            );

        }


        setTimeout(
            () => {

                error.remove();

            },
            4000
        );

    }


    /* =====================================================
       GLOBAL DRAG PREVENTION
    ===================================================== */

    document.addEventListener(
        "dragover",
        (event) => {

            event.preventDefault();

        }
    );


    document.addEventListener(
        "drop",
        (event) => {

            if (
                uploadArea &&
                uploadArea.contains(
                    event.target
                )
            ) {

                return;

            }

            event.preventDefault();

        }
    );


    /* =====================================================
       PAGE UNLOAD
    ===================================================== */

    window.addEventListener(
        "beforeunload",
        () => {

            if (cameraStream) {

                cameraStream
                    .getTracks()
                    .forEach(
                        (track) => {
                            track.stop();
                        }
                    );

            }

        }
    );


    /* =====================================================
       INITIALIZATION
    ===================================================== */

    console.log(
        "RoadGuard AI frontend initialized."
    );

    console.log(
        "Camera detection module initialized."
    );

});