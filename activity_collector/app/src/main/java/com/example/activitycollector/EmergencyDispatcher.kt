package com.example.activitycollector

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationManager
import android.net.Uri
import android.os.Build
import android.telephony.SmsManager
import android.util.Log
import androidx.core.content.ContextCompat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * EmergencyDispatcher
 * ===================
 * Dispatches emergency SOS alerts when a confirmed fall is not cancelled by the user:
 * 1. Obtains the most recent GPS location coordinates (if permitted).
 * 2. Broadcasts an emergency SMS with GPS Google Maps link to ALL registered contacts.
 * 3. Places an automated direct telephone call to the 1st (Primary) emergency contact.
 */
object EmergencyDispatcher {

    private const val TAG = "EmergencyDispatcher"

    data class DispatchResult(
        val totalContacts: Int,
        val smsSuccessCount: Int,
        val primaryCallAttempted: Boolean,
        val primaryContactName: String?,
        val locationFound: Boolean,
        val summary: String
    )

    fun dispatchEmergency(context: Context): DispatchResult {
        Log.w(TAG, "🚨 DISPATCHING EMERGENCY SOS PROTOCOL")
        val contactManager = EmergencyContactManager(context)
        val contacts = contactManager.getContacts()

        if (contacts.isEmpty()) {
            val msg = "No emergency contacts registered! Please add contacts in the app."
            Log.e(TAG, msg)
            return DispatchResult(
                totalContacts = 0,
                smsSuccessCount = 0,
                primaryCallAttempted = false,
                primaryContactName = null,
                locationFound = false,
                summary = msg
            )
        }

        // 1. Fetch current or last known location
        val location = getLastKnownLocation(context)
        val locationStr = if (location != null) {
            "https://maps.google.com/?q=${location.latitude},${location.longitude}"
        } else {
            "Location unavailable"
        }

        val timeStr = SimpleDateFormat("HH:mm:ss dd-MMM", Locale.getDefault()).format(Date())
        val smsBody = "EMERGENCY SOS: Fall detected by FallGuard! User did not respond to check-in. Time: $timeStr. Location: $locationStr. Please call or assist immediately!"

        // 2. Send SMS to ALL contacts in list
        var smsSuccessCount = 0
        val hasSmsPermission = ContextCompat.checkSelfPermission(
            context, Manifest.permission.SEND_SMS
        ) == PackageManager.PERMISSION_GRANTED

        if (hasSmsPermission) {
            try {
                val smsManager: SmsManager = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    context.getSystemService(SmsManager::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    SmsManager.getDefault()
                }

                for (contact in contacts) {
                    try {
                        val parts = smsManager.divideMessage(smsBody)
                        if (parts.size > 1) {
                            smsManager.sendMultipartTextMessage(contact.phone, null, parts, null, null)
                        } else {
                            smsManager.sendTextMessage(contact.phone, null, smsBody, null, null)
                        }
                        smsSuccessCount++
                        Log.i(TAG, "SOS SMS sent to ${contact.name} (${contact.phone})")
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed sending SMS to ${contact.name}: ${e.message}")
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing SmsManager: ${e.message}")
            }
        } else {
            Log.w(TAG, "SEND_SMS permission not granted — skipping direct SMS transmission.")
        }

        // 3. Initiate phone call to 1st (Primary) contact
        val primary = contacts.first()
        var callPlaced = false
        val hasCallPermission = ContextCompat.checkSelfPermission(
            context, Manifest.permission.CALL_PHONE
        ) == PackageManager.PERMISSION_GRANTED

        try {
            val callIntent = if (hasCallPermission) {
                Intent(Intent.ACTION_CALL, Uri.parse("tel:${primary.phone}")).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
            } else {
                Intent(Intent.ACTION_DIAL, Uri.parse("tel:${primary.phone}")).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
            }
            context.startActivity(callIntent)
            callPlaced = true
            Log.i(TAG, "Emergency call initiated to primary contact: ${primary.name} (${primary.phone})")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initiate call to primary contact: ${e.message}")
        }

        val summary = "SOS Dispatched: Call initiated to ${primary.name}. SMS sent to $smsSuccessCount of ${contacts.size} contacts."
        return DispatchResult(
            totalContacts = contacts.size,
            smsSuccessCount = smsSuccessCount,
            primaryCallAttempted = callPlaced,
            primaryContactName = primary.name,
            locationFound = location != null,
            summary = summary
        )
    }

    private fun getLastKnownLocation(context: Context): Location? {
        val hasFine = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
        val hasCoarse = ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
        if (!hasFine && !hasCoarse) return null

        return try {
            val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
            val gpsLoc = if (lm.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                lm.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            } else null

            val netLoc = if (lm.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                lm.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            } else null

            when {
                gpsLoc != null && netLoc != null -> if (gpsLoc.time > netLoc.time) gpsLoc else netLoc
                gpsLoc != null -> gpsLoc
                else -> netLoc
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not acquire location: ${e.message}")
            null
        }
    }
}
