# Highlighting Frames in Final Long-Exposure-Image

- **Status**: Accepted
- **Deciders**: [Maik Roth](https://github.com/MaikRoth)
- **Date**: 2024-05-28

## Context
As a user, I want to highlight specific frames in the final montage to draw attention to key moments. This feature is essential for enhancing the visibility of important frames, making the final long-exposure images more expressive and meaningful.

## Decision
Implement a feature that allows users to highlight specific frames in the final montage. This will be achieved through the following steps:
1. Extend the current video processing backend to support frame highlighting.
2. Modify the front-end to include buttons or a selection mechanism for highlighting frames.
3. Adjust the long-exposure image generation logic to blend highlighted frames with increased prominence.
4. Ensure the changes are well-documented and the UI is user-friendly.

## Consequences
**Positive:**
- **Enhanced User Control:** Users can emphasize important frames, making the final images more impactful.
- **Improved Visual Appeal:** Highlighted frames can enhance the storytelling aspect of the long-exposure image, drawing attention to key moments.

**Negative:**
- **Increased Complexity:** The implementation requires additional logic in both the backend and the frontend, increasing the complexity of the system.
- **UI/UX Overhead:** Adding new controls to the UI might clutter the interface if not designed carefully.

**Risks:**
- **Performance Impact:** Handling additional user inputs and processing highlights may impact performance, especially with large videos.
- **User Confusion:** If the UI is not intuitive, users might struggle to understand how to highlight frames effectively.

## Time Investment
A maximum of 8 hours will be invested in this issue. The time will be documented in a Markdown file within the repository, detailing the steps taken and time spent on each task.

## Time Log
- **Analysis & Design:** 1 hour
- **Backend Implementation:** 4 hours
- **Frontend Implementation:** 2 hours
- **Testing & Debugging:** 1 hour

## Implementation Steps
1. **Backend Modifications:**
   - Update the video processing logic to handle highlighted frames.
   - Implement a function to blend highlighted frames with the rest of the montage.

2. **Frontend Modifications:**
   - Add UI elements (buttons) to allow users to select frames for highlighting.
   - Ensure the UI is intuitive and visually appealing.

3. **Long-Exposure Image Generation:**
   - Adjust the image generation logic to blend highlighted frames with increased brightness.
