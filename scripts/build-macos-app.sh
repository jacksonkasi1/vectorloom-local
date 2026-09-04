#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$PROJECT_ROOT/dist"
APP_DIR="$DIST_DIR/VectorLoom.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
BUILD_DIR="$PROJECT_ROOT/target/macos-app"

case "$APP_DIR" in
  "$PROJECT_ROOT/dist/VectorLoom.app") rm -rf -- "$APP_DIR" ;;
  *) echo "Refusing to clean unexpected app path" >&2; exit 1 ;;
esac
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$BUILD_DIR"

cargo build --quiet --manifest-path "$PROJECT_ROOT/Cargo.toml" --release --target x86_64-apple-darwin
CARGO_PROFILE_RELEASE_OPT_LEVEL=1 CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256 \
  cargo build --quiet --manifest-path "$PROJECT_ROOT/Cargo.toml" --release --target aarch64-apple-darwin

swiftc -target x86_64-apple-macos11.0 \
  -framework AppKit -framework WebKit \
  "$PROJECT_ROOT/desktop/VectorLoomApp.swift" \
  -o "$BUILD_DIR/VectorLoom-x86_64"
swiftc -target arm64-apple-macos11.0 \
  -framework AppKit -framework WebKit \
  "$PROJECT_ROOT/desktop/VectorLoomApp.swift" \
  -o "$BUILD_DIR/VectorLoom-arm64"

lipo -create \
  "$BUILD_DIR/VectorLoom-x86_64" \
  "$BUILD_DIR/VectorLoom-arm64" \
  -output "$MACOS_DIR/VectorLoom"
lipo -create \
  "$PROJECT_ROOT/target/x86_64-apple-darwin/release/vectorloom-local" \
  "$PROJECT_ROOT/target/aarch64-apple-darwin/release/vectorloom-local" \
  -output "$RESOURCES_DIR/vectorloom-local"

cp "$PROJECT_ROOT/desktop/Info.plist" "$CONTENTS_DIR/Info.plist"
cp -R "$PROJECT_ROOT/web" "$RESOURCES_DIR/web"
chmod +x "$MACOS_DIR/VectorLoom" "$RESOURCES_DIR/vectorloom-local"
codesign --force --deep --sign - "$APP_DIR"

rm -f -- "$DIST_DIR/VectorLoom-macOS-universal.zip"
ditto -c -k --sequesterRsrc --keepParent "$APP_DIR" "$DIST_DIR/VectorLoom-macOS-universal.zip"

DMG_STAGE="$(mktemp -d "${TMPDIR:-/tmp}/vectorloom-dmg.XXXXXX")"
trap 'rm -rf -- "$DMG_STAGE"' EXIT
cp -R "$APP_DIR" "$DMG_STAGE/VectorLoom.app"
ln -s /Applications "$DMG_STAGE/Applications"
rm -f -- "$DIST_DIR/VectorLoom-macOS-universal.dmg"
hdiutil create -quiet -volname "VectorLoom" -srcfolder "$DMG_STAGE" -ov -format UDZO \
  "$DIST_DIR/VectorLoom-macOS-universal.dmg"

echo "Built $APP_DIR"
echo "Installer $DIST_DIR/VectorLoom-macOS-universal.zip"
echo "Disk image $DIST_DIR/VectorLoom-macOS-universal.dmg"
