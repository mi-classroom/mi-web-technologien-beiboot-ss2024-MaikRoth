# Excluding Irrelevant Frames or Passages in Final Montage

- **Status**: Accepted
- **Deciders**: [Maik Roth](https://github.com/MaikRoth)
- **Date**: 2024-05-28

## Context
As a user, I want to exclude irrelevant frames or passages from the final montage to have more control over the end result. This feature is crucial for enhancing the user's ability to refine and perfect their long-exposure images by removing unwanted parts of the video.

## Decision
Implement a feature that allows users to select specific frames or time segments to exclude from the final montage. This will be achieved through the following steps:
1. Extend the current video processing backend to support frame exclusion.
2. Modify the front-end to include checkboxes or a range slider for selecting frames to exclude.
3. Adjust the long-exposure image generation logic to skip excluded frames.
4. Ensure the changes are well-documented and the UI is user-friendly.

## Consequences
**Positive:**
- **Enhanced User Control:** Users can fine-tune their final images by excluding unwanted content.
- **Improved Output Quality:** The ability to remove irrelevant frames results in a more polished and desired end product.

**Negative:**
- **Increased Complexity:** The implementation requires additional logic in both the backend and the frontend, making the system more complex.
- **UI/UX Overhead:** Adding new controls to the UI might clutter the interface if not designed carefully.

**Risks:**
- **Performance Impact:** Handling additional user inputs and processing exclusions may impact performance, especially with large videos.
- **User Confusion:** If the UI is not intuitive, users might struggle to understand how to exclude frames effectively.

## Time Investment
A maximum of 16 hours will be invested in this issue. The time will be documented in a Markdown file within the repository, detailing the steps taken and time spent on each task.

## Time Log
- **Analysis & Design:** 2 hours
- **Backend Implementation:** 5 hours
- **Frontend Implementation:** 4 hours
