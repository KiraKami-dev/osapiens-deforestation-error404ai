import {
  getApp,
  getApps,
  initializeApp,
  type FirebaseApp,
  type FirebaseOptions,
} from 'firebase/app'
import { getAnalytics, isSupported, type Analytics } from 'firebase/analytics'

function getFirebaseOptions(): FirebaseOptions {
  return {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
    projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
    storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
    messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
    appId: import.meta.env.VITE_FIREBASE_APP_ID,
    measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID || undefined,
  }
}

let app: FirebaseApp | undefined

export function getFirebaseApp(): FirebaseApp {
  if (!app) {
    const options = getFirebaseOptions()
    if (
      !options.apiKey ||
      !options.authDomain ||
      !options.projectId ||
      !options.appId
    ) {
      throw new Error(
        'Firebase is not configured. Copy .env.example to .env.local and set VITE_FIREBASE_* values.',
      )
    }
    app = getApps().length > 0 ? getApp() : initializeApp(options)
  }
  return app
}

let analyticsPromise: Promise<Analytics | null> | null = null

export function getFirebaseAnalytics(): Promise<Analytics | null> {
  if (!analyticsPromise) {
    analyticsPromise = isSupported().then((supported) =>
      supported ? getAnalytics(getFirebaseApp()) : null,
    )
  }
  return analyticsPromise
}

/** Call once at startup (e.g. from main.tsx). */
export function initFirebase(): FirebaseApp {
  const firebaseApp = getFirebaseApp()
  void getFirebaseAnalytics()
  return firebaseApp
}
