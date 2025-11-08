# Code Review: PR #13 - UI Dashboard

**Reviewer**: Claude
**Date**: 2025-11-08
**PR Author**: j-d-p-c
**Branch**: `ui` → `main`

## Summary

This PR introduces a Streamlit-based dashboard for animal detection and counting in aerial images. The implementation includes image upload capabilities, S3 integration, API client utilities, and visualization tools for detection results. The dashboard temporarily uses Faster R-CNN detection results to simulate HerdNet centroids.

**Files Changed**: 12 files (+690, -5 lines)

## Overall Assessment

**Status**: ⚠️ CHANGES REQUESTED

The PR provides good foundational functionality for the dashboard, but several issues need to be addressed before merging, particularly around error handling, security, configuration management, and code consistency.

---

## Critical Issues 🔴

### 1. Missing Timeout on API Requests
**Location**: `dashboard/utils/api_client.py`

Most API requests lack timeout parameters, which can cause the application to hang indefinitely.

```python
# Current (problematic):
response = requests.get(endpoint)

# Should be:
response = requests.get(endpoint, timeout=30)
```

**Impact**: Application can freeze indefinitely if backend is unresponsive.

**Files affected**:
- `get_regions()` - line 18
- `get_flyovers()` - line 28
- `get_detection_results()` - line 38
- `get_counts_for_flyover()` - line 60
- `get_counts_for_region()` - line 71

**Recommendation**: Add consistent timeout values (e.g., 30 seconds for GET, 300 for POST with heavy processing).

### 2. Hardcoded Configuration Values
**Locations**: Multiple files

Critical configuration is hardcoded rather than using environment variables:

```python
# dashboard/utils/s3_utils.py:9
S3_BUCKET_NAME = "cow-detect-maia"  # Should be env var

# dashboard/utils/api_client.py:48
timeout=300  # Should be configurable
```

**Impact**:
- Cannot easily change between dev/staging/prod environments
- Security risk if bucket name should be private
- Difficult to test with different configurations

**Recommendation**:
- Use environment variables with reasonable defaults
- Consider a central config file or `.env` file
- Document all required environment variables

### 3. No Input Validation
**Location**: `dashboard/pages/1_Upload_Images.py`

User inputs (farm_name, dates, files) are not validated before processing:

```python
farm_name = st.text_input("Etiqueta", value="Prueba")  # No validation
# Could contain special characters that break S3 paths or SQL injection attempts
```

**Impact**:
- Potential for injection attacks
- S3 path conflicts if special characters used
- Invalid data in database

**Recommendation**:
- Add input sanitization for `farm_name` (alphanumeric + limited special chars)
- Validate file types beyond extension (check magic bytes)
- Validate date ranges are reasonable

### 4. AWS Credentials Error Handling
**Location**: `dashboard/utils/s3_utils.py:18-23`

The S3 client creation only catches `NoCredentialsError`, but other AWS errors can occur:

```python
try:
    return boto3.client("s3")
except NoCredentialsError:  # Too narrow
    st.error("Error de credenciales de AWS...")
    return None
```

**Impact**: Uncaught exceptions for permission errors, region misconfigurations, etc.

**Recommendation**: Catch broader `ClientError` and provide specific error messages.

---

## High Priority Issues 🟡

### 5. Inconsistent Error Handling
**Location**: `dashboard/utils/api_client.py`

Functions return different error formats, making it hard to handle errors consistently:

```python
# Some return empty list on error:
return []

# Others return dict with error key:
return {"error": str(e)}
```

**Impact**: Calling code must handle multiple error patterns.

**Recommendation**: Standardize error handling:
- Return consistent error structure
- Consider raising exceptions and handling in UI layer
- Log errors for debugging

### 6. Unsafe Type Casting
**Location**: `dashboard/utils/api_client.py`

Using `cast()` to silence type checkers without runtime validation:

```python
return cast(list[str], response.json().get("regions", []))
```

**Impact**: If API returns unexpected type, errors occur at runtime instead of being caught early.

**Recommendation**:
- Add runtime validation using Pydantic models
- Use proper type guards
- Don't use `cast()` as a shortcut

### 7. File Upload Without Cleanup
**Location**: `dashboard/pages/1_Upload_Images.py`

No mechanism to clean up uploaded files from S3 if processing fails:

```python
s3_urls = upload_files_to_s3(...)  # Files uploaded
results = process_images(image_urls=s3_urls)  # If this fails, files remain in S3
if "error" in results:
    st.error(...)  # No cleanup!
```

**Impact**: S3 storage fills with orphaned files, increasing costs.

**Recommendation**:
- Implement rollback mechanism
- Add lifecycle policies to S3 bucket
- Track uploaded files and provide cleanup utility

### 8. Aggressive Caching Without Invalidation
**Location**: `dashboard/utils/visualization.py:41`

`@st.cache_data` on `download_image()` with no TTL or invalidation:

```python
@st.cache_data
def download_image(url: str) -> Image.Image | None:
```

**Impact**:
- If image is updated in S3, old version shown
- Memory usage grows unbounded
- No way to clear cache

**Recommendation**:
- Add TTL parameter
- Add cache size limits
- Provide cache clear mechanism

### 9. Missing Device Selection in API
**Location**: `api/model_utils.py:102-107`

Device selection logic only checks for MPS, no GPU (CUDA) support:

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

**Impact**: Won't use CUDA GPU even if available, resulting in slower inference.

**Recommendation**:
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

---

## Medium Priority Issues 🟠

### 10. Language Inconsistency
**Location**: Throughout codebase

Mix of Spanish and English in code, comments, and UI:

```python
# Spanish in code:
def get_regions() -> list[str]:
    """Obtiene la lista de regiones disponibles desde la API."""

# English variable names but Spanish UI:
st.title("Carga y Procesamiento de Imágenes")
```

**Impact**:
- Reduces code readability for international contributors
- Makes maintenance harder
- Inconsistent user experience

**Recommendation**:
- Choose one language for code/comments (typically English)
- Use i18n for UI strings to support multiple languages
- Document language policy in CONTRIBUTING.md

### 11. Missing Type Hints
**Location**: `api/animaloc_utils.py:159`

The hardcoded batch_size change lacks explanation in type hints or return type:

```python
batch_size=1,  # cfg.inference_settings.batch_size,
```

**Impact**: Developer might not understand why batch_size > 1 is commented out.

**Recommendation**: Add TODO or link to issue explaining limitation.

### 12. Font Fallback May Fail Silently
**Location**: `dashboard/utils/visualization.py:101-104`

Font loading falls back to default without informing user:

```python
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
except OSError:
    font = ImageFont.load_default()
```

**Impact**: UI may look unprofessional on systems without DejaVuSans.

**Recommendation**:
- Package font with dashboard or specify installation requirement
- Log warning when falling back to default font
- Consider using system fonts

### 13. Progress Bar Not Always Cleared
**Location**: `dashboard/utils/s3_utils.py:40-54`

Progress bar might not clear if exception occurs mid-upload:

```python
for i, file in enumerate(files):
    try:
        # ... upload code ...
    except Exception as e:
        st.error(...)
        progress_bar.empty()  # Only cleared on error, not on success
        return None

progress_bar.empty()  # Never reached if exception
```

**Impact**: Lingering UI elements confuse users.

**Recommendation**: Use try/finally or context manager to ensure cleanup.

### 14. Magic Numbers
**Location**: Multiple files

Various hardcoded values without named constants:

```python
# dashboard/utils/visualization.py:103
font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)  # Why 20?

# dashboard/utils/api_client.py:48
timeout=300  # Why 300?
```

**Recommendation**: Define named constants at module level with explanatory comments.

---

## Low Priority / Nice-to-Have 🔵

### 15. Missing Dependencies in Type Checking
**Location**: `pyproject.toml`

Added `boto3` but not `boto3-stubs[s3]` for type checking:

```toml
[project.optional-dependencies]
dashboard = [
    "boto3>=1.34.0",  # Missing type stubs
]
```

**Recommendation**: Add `boto3-stubs[s3]` for better IDE support and type safety.

### 16. No Loading State for Long Operations
**Location**: `dashboard/pages/2_View_Detections.py`

Image download happens without loading indicator (except for cached results):

```python
image = download_image(selected_image_url)  # Could be slow
```

**Recommendation**: Add `st.spinner()` for first-time downloads.

### 17. URL Display in Selectbox
**Location**: `dashboard/pages/2_View_Detections.py:134-137`

URL selectbox shows full S3 paths which are hard to read:

```python
format_func=lambda url: url.split("/")[-1]  # Only shows filename
```

**Impact**: User can't distinguish between identically named files from different uploads.

**Recommendation**: Show more context like `{date}/{filename}` or add index numbers.

### 18. Limited Error Context
**Location**: `dashboard/utils/api_client.py`

Error messages don't include response status codes or response body:

```python
except requests.exceptions.RequestException as e:
    st.error(f"Error al obtener regiones: {e}")
```

**Recommendation**: Include `response.status_code` and `response.text` for debugging.

---

## Positive Aspects ✅

1. **Good Architecture**: Clear separation of concerns (API client, S3, visualization)
2. **Type Hints**: Most functions have proper type annotations
3. **Documentation**: Good docstrings explaining function purposes
4. **User Experience**: Progressive disclosure with sidebar controls
5. **Visualization Options**: Flexible display modes for different use cases
6. **Progress Indicators**: Upload progress shown to users
7. **Config File**: Streamlit theming properly configured
8. **Bug Fixes**: Fixed important typos (`datasets` → `dataset`) in existing code

---

## Testing Recommendations

### Unit Tests Needed:
1. `api_client.py`: Mock API responses, test error handling
2. `s3_utils.py`: Mock boto3 client, test upload/failure scenarios
3. `visualization.py`: Test drawing functions with sample data

### Integration Tests Needed:
1. End-to-end upload and visualization flow
2. API timeout scenarios
3. S3 permission errors

### Manual Testing Checklist:
- [ ] Upload single image
- [ ] Upload multiple images (>10)
- [ ] Test with missing AWS credentials
- [ ] Test with backend API down
- [ ] Test with invalid image files
- [ ] Test filtering by species
- [ ] Test all visualization modes
- [ ] Test with different confidence thresholds
- [ ] Test on systems without DejaVuSans font

---

## Security Considerations

1. **Input Sanitization**: Add validation for all user inputs
2. **S3 Bucket Permissions**: Ensure bucket policy restricts access appropriately
3. **API Authentication**: Consider adding auth tokens for API requests
4. **CORS**: Verify CORS policies if dashboard deployed separately from API
5. **Secrets Management**: Use AWS Secrets Manager or similar for credentials

---

## Recommendations for Merge

### Must Fix Before Merge:
1. ✅ Add timeouts to all API requests (#1)
2. ✅ Move configuration to environment variables (#2)
3. ✅ Add input validation (#3)
4. ✅ Fix device selection to include CUDA (#9)
5. ✅ Standardize error handling (#5)

### Should Fix Before Merge:
6. ⚠️ Implement S3 cleanup mechanism (#7)
7. ⚠️ Add cache invalidation strategy (#8)
8. ⚠️ Improve AWS error handling (#4)
9. ⚠️ Address language inconsistency (#10)

### Can Fix in Follow-up PR:
10. 📝 Add comprehensive tests
11. 📝 Add type stubs for boto3
12. 📝 Improve error messages with more context
13. 📝 Extract magic numbers to constants

---

## Suggested Changes

### Configuration Management
Create a `dashboard/config.py`:

```python
import os
from dataclasses import dataclass

@dataclass
class DashboardConfig:
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    s3_bucket_name: str = os.getenv("S3_BUCKET_NAME", "cow-detect-maia")
    api_timeout: int = int(os.getenv("API_TIMEOUT", "30"))
    api_timeout_heavy: int = int(os.getenv("API_TIMEOUT_HEAVY", "300"))
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))

config = DashboardConfig()
```

### Error Response Standard
Create a `dashboard/utils/errors.py`:

```python
from typing import TypedDict

class APIError(TypedDict):
    error: str
    status_code: int | None
    detail: str | None

def handle_api_error(e: Exception, response=None) -> APIError:
    return {
        "error": str(e),
        "status_code": response.status_code if response else None,
        "detail": response.text if response else None,
    }
```

### Input Validation
Add to `dashboard/utils/validation.py`:

```python
import re
from pathlib import Path

def validate_region_name(name: str) -> str:
    """Validate and sanitize region/farm name."""
    if not name:
        raise ValueError("Region name cannot be empty")

    # Allow only alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValueError("Region name contains invalid characters")

    if len(name) > 100:
        raise ValueError("Region name too long")

    return name

def validate_image_file(file) -> bool:
    """Validate uploaded file is actually an image."""
    # Check magic bytes, not just extension
    allowed_types = {
        b'\xFF\xD8\xFF': 'jpg',
        b'\x89\x50\x4E\x47': 'png',
    }

    file.seek(0)
    header = file.read(4)
    file.seek(0)

    return any(header.startswith(magic) for magic in allowed_types.keys())
```

---

## Conclusion

This PR provides a solid foundation for the dashboard interface. The code is well-structured and demonstrates good software engineering practices in several areas. However, there are important issues around error handling, configuration management, and security that should be addressed before merging.

The most critical issues (#1-4) are relatively straightforward to fix and would significantly improve the robustness and maintainability of the application.

**Estimated effort to address critical issues**: 2-4 hours
**Recommended action**: Request changes, provide this feedback to author

---

## References

- [Streamlit Best Practices](https://docs.streamlit.io/library/advanced-features/configuration)
- [Boto3 Error Handling](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html)
- [Requests Timeouts](https://requests.readthedocs.io/en/latest/user/advanced/#timeouts)
- [OWASP Input Validation](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html)
