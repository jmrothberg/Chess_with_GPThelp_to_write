# Fix: Menu Bar Disappearing in Cursor IDE (GNOME)

## Problem
Menu bar disappears in Cursor IDE after Claude updates, especially on GNOME desktop environment.

## Root Cause
GNOME's global menu or app menu integration conflicts with certain Electron apps like Cursor.

## PRIORITY: Cursor-Specific JSON Fixes (Most Likely to Work)

### Method 1: Cursor Settings.json
Open Cursor → File → Preferences → Settings (JSON)

Add to your `settings.json`:
```json
{
  "window.menuBarVisibility": "visible",
  "window.nativeTabs": false,
  "window.titleBarStyle": "custom"
}
```

### Method 2: Cursor Command Line Flags
Create or edit `~/.config/Cursor/User/settings.json`:
```json
{
  "window.menuBarVisibility": "visible",
  "window.titleBarStyle": "custom",
  "application.menuBarVisibility": "visible"
}
```

### Method 3: Launch with Flags
Create desktop shortcut or modify launcher:
```bash
cursor --gtk-version=3 --disable-features=VizDisplayCompositor
```

## Alternative GNOME Fixes

### Method 4: Environment Variable
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export GTK_CSD=0
export XDG_CURRENT_DESKTOP=GNOME
```

### Method 5: GNOME dconf Settings
```bash
# Check current settings
gsettings get org.gnome.desktop.wm.preferences button-layout

# Override window decorations
gsettings set org.gnome.desktop.wm.preferences button-layout ':minimize,maximize,close'
```

### Method 6: Cursor Config File
Create `~/.config/cursor-flags.conf`:
```
--gtk-version=3
--disable-features=VizDisplayCompositor
--enable-features=UseOzonePlatform
--ozone-platform=x11
```

## Quick Test
1. Apply fix
2. **Fully restart Cursor** (not just reload window)
3. Check if menu bar appears at top of window

## When This Happens
- After Claude/Cursor updates
- After system updates
- When switching between different desktop environments

## Emergency Fix
If nothing works, try running Cursor from terminal:
```bash
cursor --verbose --disable-gpu 2>&1 | grep -i menu
```
